import os
import json
import re
import time
from dotenv import load_dotenv
import openai

# === Загрузка переменных окружения ===
load_dotenv()

# === Автоматическое определение директории скрипта ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

client = openai.OpenAI(
    api_key=os.getenv("SECRET_TOKEN"),
    base_url=os.getenv("MODEL_API_BASE"),
)

# === Маппинг Bloom-типов по порядку (state[3] → bloom_id) ===
BLOOM_TYPE_MAP = {
    1:  "Verification question in which some information is provided and it is necessary to determine whether it is correct or not",
    2:  "Classification task: assign each given term to its correct category based on memorized facts",
    3:  "Constructed-response question providing no hints or accompanying information",
    4:  "Fill-in-the-blank question/task providing several hints",
    5:  "Constructed-response question where information is presented in one form and must be represented in another form",
    6:  "Constructed-response question requiring an example",
    7:  "Constructed-response question providing a specific example where it is necessary to identify the corresponding concept or principle from a given list",
    8:  "Sorting question/task providing a set of examples/objects where it is necessary to determine which belong to the specified category and which do not, OR assign each example/object to one of several categories",
    9:  "Constructed-response question related either to themes or to summaries/abstracts",
    10: "Completion question/task where the user is given a series of elements and must determine what comes next",
    11: "Hierarchical-analogy task: given a part-whole or category-example pair, complete the parallel pair",
    12: "Odd-one-out question/task where the user is given three or more items and must determine which one does not belong",
    13: "Matching question/task requiring demonstration of how each part of one object, idea, problem, or situation corresponds to (or maps onto) each part of another",
    14: "Reasoning question/task asking you to explain the cause of a given event",
    15: "Troubleshooting question/task asking the user to determine what might have gone wrong in a malfunctioning system",
    16: "System modification question/task asking the user to modify a system to achieve a certain goal",
    17: "Prediction question/task asking how a change in one part of a system will affect another part",
    18: "Algorithm execution question/task presenting a familiar problem solvable by a known procedure",
    19: "Algorithm execution question/task presenting an unfamiliar problem that must be solved",
    20: "Sequencing task: arrange the given elements in the correct factual order",
    21: "Pattern task: given 2-3 factual examples, name the shared principle they illustrate",
    22: "Analogy-interpretation task: given a complete analogy, state the shared logical relation",
}


def get_bloom_id(state: list) -> int | None:
    """
    Определяет bloom_id по полю state[3] через строгое сопоставление подстроки.
    """
    if not state or len(state) < 4:
        return None
    descriptor = state[3]
    for bloom_id, keyword in BLOOM_TYPE_MAP.items():
        if keyword in descriptor:
            return bloom_id
    return None


def extract_item_fields(item: dict) -> tuple[str | None, str | None]:
    """
    Извлекает question и answer из элемента датасета.
    """
    if item.get("success", True):
        question = item.get("question")
        answer = item.get("answer")
    else:
        out = item.get("out", {})
        question = out.get("question") if isinstance(out, dict) else None
        answer = None
    return question, answer


def generateLLMResponse(prompt):
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=60.0
    )
    time.sleep(1)
    return response.choices[0].message.content


def check_answerability(question: str, domain: str = "естественные науки"):
    """
    Оценивает отвечаемость вопроса через LLM.
    """
    
    prompt = f"""You are a strict test quality auditor and logic expert.
Your task: Evaluate the "Answerability" of the proposed question in the domain "{domain}".
"Answerability" means the question is formulated such that there exists an objectively correct,
logical, and complete answer that can be given based EXCLUSIVELY on the question text (and
generally known domain facts), without needing to guess the author's intentions.

VERIFICATION CRITERIA:
1. Contextual self-sufficiency: Is the object of the question clear? (Bad: "Почему он не выживет?" — it's unclear who "he" is).
2. Completeness of conditions: Is there enough data to solve? (Bad: "Определите класс растения, если известно, что у него есть ядро" (it's not enough to define class)).
3. Objectivity: Does a factual answer exist? (Bad: "Какое хвойное растение лучше всех пахнет?").
4. Self-sufficiency: Are there no references to missing content ("согласно данному определению", "в тексте выше"), but there is no text before?
5. Logical integrity: Are there no contradictions in the conditions?


ANSWERABILITY SCALE (0–4):

SCORE 0: IMPOSSIBLE / NONSENSE (Cannot be answered)
- The question is a set of words, a sentence fragment, or nonsense.
- The subject of the question is absent.
- Example: "Потому что растение в лесу?", "Код ошибка сервер".

SCORE 1: CRITICAL DEFICIT / PURELY SUBJECTIVE (Critical defect)
- Key data/code/context is missing, making a solution impossible in principle.
- The question refers to a missing object ("В приведенном выше тексте...", "согласно данному определению" — but there is no text given before).
- The question requires pure opinion/taste without criteria ("Что лучше?").
- Example: "Что это за растение?" (the plant is not specified).

SCORE 2: AMBIGUOUS / GUESSWORK (Ambiguity / Guessing)
- Strong assumptions are required to answer (so the answer must start like this: "наверное, имеется в виду стандартный случай" etc.).
- Grammatical errors distort the meaning.
- There is almost no description which helps to answer a very broad question.
- Example: "Как определить класс?" (It's not specified of what: plant, animal?).

SCORE 3: MINOR ISSUES / IMPLICIT CONTEXT / SEVERAL POSSIBLE ANSWERS (Minor flaws)
- The question is answerable, but the formulation is not ideal — Context is understood from generally accepted domain norms, although not explicitly stated (for example, "Напишите классификацию для сосны" — it's clear that kingdom, phylum, class, etc. are meant).
- A correct answer can be given, but it may be incomplete due to the formulation.
- There are several correct answers (e.g., "Как изменить работу нервной ткани, чтобы усилить скорость передачи нервных импульсов в организме?" — there are several ways to do this, but it's ok, because any correct answer will be acceptable in this type of questions).

SCORE 4: PERFECTLY ANSWERABLE (Fully answerable)
- The question is self-sufficient, precise, and correct.
- All variables are defined, context is clear.
- There is one or a finite set of correct answers.
- IMPORTANT: It's OK, if answering the question REQUIRES SPECIFIC KNOWLEDGE in the different fields of/near "{domain}", including specific (and even rare) facts, terms, and concepts (and this knowledge is sufficient for a complete correct answer).


RESPONSE FORMAT — STRICTLY JSON:
{{
  "answerability_score": <int 0..4>,
  "verdict": "Impossible" | "Deficient" | "Ambiguous" | "Good" | "Perfect",
  "issue_type": "None" | "Missing Context" | "Missing Data" | "Subjective" | "Logical Error" | "Several Possible Answers" | "Grammar",
  "missing_elements": ["element1", "element2"] or "None",
  "reasoning": "Brief explanation IN RUSSIAN. Why can't it be scored 4? What exactly is missing? If score is 4, write OK.",
  "improvement_suggestion": "Rewrite the question in the SAME format (question type) IN RUSSIAN so that it is fully answerable (if score < 4), else OK."
}}

Question to evaluate:
<<<QUESTION
{question}
QUESTION>>>"""

    def _try_parse(raw):
        # Строгий и надежный поиск JSON-блока: от первой { до последней }
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        return json.loads(raw.strip())

    for attempt in range(2):
        try:
            raw_response = generateLLMResponse(prompt)
            return _try_parse(raw_response)
        except json.JSONDecodeError as e:
            if attempt == 0:
                print(f"Ошибка парсинга JSON: {e}, повтор...")
            else:
                return None
        except Exception as e:
            if attempt == 0:
                print(f"Ошибка при чекинге: {e}, повтор...")
            else:
                return None
    return None


def check_instructions_file(filepath: str, domain: str = "естественные науки", output_filepath: str = None):
    """
    Проверяет все элементы из файла с инструкциями на отвечаемость.
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(SCRIPT_DIR, filepath)
    if output_filepath and not os.path.isabs(output_filepath):
        output_filepath = os.path.join(SCRIPT_DIR, output_filepath)

    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}

    for grade, instructions in data.items():
        results[grade] = []
        for item in instructions:
            item_id = item.get("id", "unknown")
            success = item.get("success", True)
            question, answer = extract_item_fields(item)

            # Определяем bloom_id и берем остальные поля
            bloom_id = item.get("bloom_id") or get_bloom_id(item.get("state", []))
            state = item.get("state", [])
            definition = item.get("definition")

            # === ОТБРАКОВКА ДО LLM И ДО ПРОВЕРКИ ВОПРОСА ===
            if not success:
                check_result = {
                    "answerability_score": 0,
                    "verdict": "Impossible",
                    "issue_type": "None",
                    "missing_elements": "None",
                    "reasoning": "NOTHING",
                    "improvement_suggestion": "NOTHING"
                }
                print(f"💥 [{grade}][id={item_id}] Отбракован (success=False): Score 0/4")
            else:
                # Успешные проверяем на наличие вопроса
                if not question:
                    print(f"[{grade}][id={item_id}] Пропущен: нет вопроса")
                    continue

                # Успешные отправляем на фильтрацию LLM
                check_result = check_answerability(
                    question=question,
                    domain=domain
                )
                
                if not check_result:
                    print(f"❌ [{grade}][id={item_id}] Ошибка LLM, пропуск")
                    continue

                status_emoji = {4: '✅', 3: '🟡', 2: '🟠', 1: '🔴', 0: '💥'}.get(
                    check_result.get("answerability_score", 2), '❓'
                )
                print(f"{status_emoji} [{grade}][id={item_id}] "
                      f"Score: {check_result.get('answerability_score')}/4 — {check_result.get('verdict')}")

            # Формируем итоговый объект (check_result всегда определен на этом этапе)
            filtered_item = {
                "id": item_id,
                "bloom_id": bloom_id,
                "success": success,
                "state": state,
                "definition": definition,
                "question": question,
                "answer": answer,
                "answerability_score": check_result.get("answerability_score"),
                "verdict": check_result.get("verdict"),
                "issue_type": check_result.get("issue_type"),
                "missing_elements": check_result.get("missing_elements"),
                "reasoning": check_result.get("reasoning"),
                "improvement_suggestion": check_result.get("improvement_suggestion"),
            }
            results[grade].append(filtered_item)

    if output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в: {output_filepath}")

    return results


if __name__ == "__main__":
    print(f"Директория скрипта: {SCRIPT_DIR}")
    print(f"Запуск проверки отвечаемости вопросов...\n")

    check_instructions_file(
        filepath="instructions_chemistry89.json",
        domain="химия",
        output_filepath="instructions_chemistry89_ans_checked.json"
    )

    print("\nПроверка завершена!")