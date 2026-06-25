import os
import json
import re
from dotenv import load_dotenv
import openai
import time

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


def fact_check_statement(question: str, answer: str | None, domain: str = "естественные науки"):
    """
    Проверяет фактологическую достоверность ответа и его соответствие вопросу.
    """
    input_block = f"""Question:
<<<QUESTION
{question}
QUESTION>>>

Answer:
<<<ANSWER
{answer}
ANSWER>>>"""

    prompt = f"""You are a strict scientific editor and Fact-Checker in the subject domain: {domain}.
Your task: Check the factual accuracy (Factual Accuracy) of the proposed question-answer pair.

You must act as a "hallucination detector".
Your goal is to find ANY deviation from the generally accepted scientific or technical consensus in the domain {domain}.

VERIFICATION CRITERIA:
1. Truthfulness: Does the answer correspond to objective reality?
2. Correspondence: Does the answer actually answer the question (not something adjacent)?
3. Terminology: Are terms used correctly?
4. Absence of hallucinations: Are facts, libraries, historical events, or physical laws not made up?


VERIFICATION CRITERIA:
1. Truthfulness: Does the answer correspond to objective reality?
2. Correspondence: Does the answer actually answer the question (not something adjacent)?
3. Terminology: Are terms used correctly (is a "method" not called a "class" if it matters)?
4. Absence of hallucinations: Are facts, libraries, historical events, or physical laws not made up?

RELIABILITY SCALE (0–4):

SCORE 0: HALLUCINATION / FABRICATION
- The statement is completely false.
- Non-existent objects, functions, laws are mentioned.
- A gross error that shows complete misunderstanding of the subject.
- Example: "Столица Франции — Берлин", "Фотосинтез происходит в митохондриях".

SCORE 1: MAJOR MISCONCEPTION
- The statement contains a grain of truth, but conclusions are drawn incorrectly.
- Cause and effect are confused.
- Answer that does not address the question.
- Gross error in key numbers/dates/formulas.
- Mixing concepts from different domains, leading to a false conclusion.
- Example: "Все бактерии вредны для человека", "Эволюция — это линейный процесс от простого к сложному".

SCORE 2: MIXED / SIGNIFICANT INACCURACY
- Part of the statement is correct, part is not.
- Critical omissions of important exceptions ("Always do X", although X sometimes breaks the system).
- The answer EXPLICITLY STATES that the answer cannot be given due to lack of data or another reason, and then only assumptions follow.
- Example: "Все млекопитающие рождают живых детенышей" (oviparous animals not included: platypus, echidna).

SCORE 3: MOSTLY CORRECT / MINOR NITPICK
- Factually correct, but there is terminological carelessness.
- The statement is correct but not entirely complete (a rare edge case is omitted).
- A small numerical error that does not affect the essence.
- Example: "Функции гемоглобина: гемоглобин переносит кислород в крови" (true, but what is not mentioned is that it also transports CO₂ and is involved in the buffering system).

SCORE 4: FACTUALLY PERFECT
- The statement is absolutely true according to the current consensus in {domain}.
- Precise, complete, and correct.
- Even a strict expert cannot find fault.


RESPONSE FORMAT — STRICTLY JSON:
{{
  "accuracy_score": <int 0..4>,
  "verdict": "Hallucination" | "False" | "Mixed" | "Mostly Correct" | "Perfect",
  "error_type": "None" | "Outdated Info" | "Hallucination" | "Terminological Error" | "Logical Fallacy" | "Off-Topic Answer" | "Incomplete Answer",
  "fact_check_details": "Quote of the erroneous fragment (or null)",
  "reasoning": "Explanation IN RUSSIAN: why is this incorrect? What is the correct fact? If everything is correct, write OK",
  "correction": "Correct the answer in RUSSIAN so that it becomes factually accurate and fully addresses the question (if score < 4, otherwise write OK)"
}}

{input_block}"""

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
    Проверяет все элементы из файла с инструкциями на фактологическую точность.
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
                    "accuracy_score": 0,
                    "verdict": "Hallucination",
                    "error_type": "None",
                    "fact_check_details": "NOTHING",
                    "reasoning": "NOTHING",
                    "correction": "NOTHING"
                }
                print(f"💥 [{grade}][id={item_id}] Отбракован (success=False): Score 0/4")
            else:
                # Успешные проверяем на наличие вопроса
                if not question:
                    print(f"[{grade}][id={item_id}] Пропущен: нет вопроса")
                    continue

                # Успешные отправляем на фильтрацию LLM
                check_result = fact_check_statement(
                    question=question,
                    answer=answer,
                    domain=domain
                )
                
                if not check_result:
                    print(f"❌ [{grade}][id={item_id}] Ошибка LLM, пропуск")
                    continue

                status_emoji = {4: '✅', 3: '🟡', 2: '🟠', 1: '🔴', 0: '💥'}.get(
                    check_result.get("accuracy_score", 2), '❓'
                )
                print(f"{status_emoji} [{grade}][id={item_id}] "
                      f"Score: {check_result.get('accuracy_score')}/4 — {check_result.get('verdict')}")

            # Формируем итоговый объект (check_result всегда определен на этом этапе)
            filtered_item = {
                "id": item_id,
                "bloom_id": bloom_id,
                "success": success,
                "state": state,
                "definition": definition,
                "question": question,
                "answer": answer,
                "accuracy_score": check_result.get("accuracy_score"),
                "verdict": check_result.get("verdict"),
                "error_type": check_result.get("error_type"),
                "reasoning": check_result.get("reasoning"),
                "correction": check_result.get("correction"),
            }
            results[grade].append(filtered_item)

    if output_filepath:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nРезультаты сохранены в: {output_filepath}")

    return results


if __name__ == "__main__":
    print(f"Директория скрипта: {SCRIPT_DIR}")
    print(f"Запуск проверки фактологической точности...\n")

    check_instructions_file(
        filepath="instructions_physics_10-11.json",
        domain="физика",
        output_filepath="instructions_physics1011_fact_checked.json"
    )

    print("\nПроверка завершена!")