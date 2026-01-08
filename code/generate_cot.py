import os
import json
import time
import openai
from tqdm import tqdm
import regex as re
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import prompt_cot

client = openai.OpenAI(
    api_key='your_api_key',
    base_url='your_base_url'
)

def call_llm(messages, model_name, temperature=0.0, sleep_time=1.0):
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            time.sleep(sleep_time)
            continue

def get_history_text(history_messages):
    history = []
    for hm in history_messages:
        if hm['role'] == 'user':
            history.append(f"来访者：{hm['content']}")
        else:
            history.append(f"咨询师：{hm['content']}")
    return '\n'.join(history)

def get_memory_text(emotion):
    return (
        f"当前情绪：{'，'.join(emotion['current_emotion'])}\n"
        f"当前情绪原因：{emotion['current_analysis']}\n"
        f"近期转变：{emotion['recent_change'] or '无'}\n"
        f"整体趋势：{emotion['overall_trend'] or '无'}\n"
        f"情绪转变原因：{emotion['shift_analysis'] or '无'}"
    )

def get_counselor_text(counselor):
    return (
        f"当前阶段：{counselor['phase']}\n"
        f"使用策略：{counselor['strategy']}"
    )

def get_safety_text(counselor_response, seeker_utterances, safety_res):
    output = []
    output.append(f'如果我（咨询师）回复：{counselor_response}')
    output.append('来访者可能的后续表述及分析：')
    for item in safety_res.get("safety_analysis", []):
        uid = item["uid"]
        analysis = item["analysis"]
        is_safe = item["pass"]
        utterance = next((u["utterance"] for idx, u in enumerate(seeker_utterances) if idx+1 == uid), "")
        output.append(f'表述：{utterance}')
        output.append(f'分析：{analysis}{"（不存在风险）" if is_safe else "（存在风险）"}\n')
    if safety_res["pass_flag"] == False:
        suggestion_emotion = safety_res['suggestion'].get('emotion', '')
        suggestion_safety = safety_res['suggestion'].get('safety', '')
        output.append(f'情绪修改建议：{suggestion_emotion}')
        output.append(f'安全修改建议：{suggestion_safety}')
    return "\n".join(output)

def generate_cot(data):
    records = data["result"]
    history_messages = []
    cot_data = {
        "id": data["id"],
        "topic": data["topic"],
        "history": []
    }

    i = 0
    try:
        while i < len(records):

            history = get_history_text(history_messages)

            if (
                i + 4 < len(records)
                and records[i]["module"] == "seeker"
                and records[i+1]["module"] == "emotion"
                and records[i+2]["module"] == "counselor"
                and records[i+3]["module"] == "simulate"
                and records[i+4]["module"] == "safety"
            ):
                seeker = records[i]
                emotion = records[i+1]
                counselor = records[i+2]
                simulate = records[i+3]
                safety = records[i+4]

                seeker_response = seeker["response"]

                cot_data["history"].append({"role": "来访者", "content": seeker_response})
                history_messages.append({"role": "user", "content": seeker_response})

                emotion_think = emotion['thinking']
                emotion_text = get_memory_text(emotion)

                i = i + 2

                safeties = []

                while (
                    i + 2 < len(records)
                    and records[i]["module"] == "counselor"
                    and records[i+1]["module"] == "simulate"
                    and records[i+2]["module"] == "safety"
                    and records[i+2]["pass_flag"] is False
                ):
                    counselor = records[i]
                    simulate = records[i+1]
                    safety = records[i+2]

                    counselor_response = counselor["response"]

                    seeker_utterances = [{"uid": idx+1, "utterance": u.get("response","")} for idx, u in enumerate(simulate["utterances"])]
                    safeties.append(get_safety_text(counselor_response, seeker_utterances, safety))

                    i += 3

                if (
                    i + 2 < len(records)
                    and records[i]["module"] == "counselor"
                    and records[i+1]["module"] == "simulate"
                    and records[i+2]["module"] == "safety"
                    and records[i+2]["pass_flag"] is True
                ):
                    counselor = records[i]
                    simulate = records[i+1]
                    safety = records[i+2]

                    counselor_think = counselor["thinking"]
                    counselor_text = get_counselor_text(counselor)
                    counselor_response = counselor["response"]

                    seeker_utterances = [{"uid": idx+1, "utterance": u.get("response","")} for idx, u in enumerate(simulate["utterances"])]
                    safeties.append(get_safety_text(counselor_response, seeker_utterances, safety))

                    safety_text = '\n\n'.join(safeties)

                    user_cot = prompt_cot.replace('{{history}}', history).replace('{{seeker_response}}', seeker_response).replace('{{emotion_think}}', emotion_think).replace('{{emotion_text}}', emotion_text).replace('{{counselor_think}}', counselor_think).replace('{{counselor_text}}', counselor_text).replace('{{safety_text}}', safety_text).replace('{{counselor_response}}', counselor_response)
                    messages_cot = [{"role": "user", "content": user_cot}]
                    while True:
                        try:
                            cot_response = call_llm(messages_cot, 'gpt-4.1-mini')
                            if len(cot_response.strip()) >= 200:
                                break
                        except:
                            continue
                    counselor_content = f"<think>\n{cot_response}\n</think>\n\n{counselor_response}"
                    cot_data["history"].append({"role": "咨询师", "content": counselor_content})
                    history_messages.append({"role": "assistant","content": counselor_response})
                    i += 3
            else:
                i += 1
        return cot_data
    except:
        return None

def run_with_pool(
    input_file: str,
    output_file: str,
    task_fn,
    *,
    key_field: str = None,
    max_workers: int = 8,
    backup_interval: int = 0,
    task_args: tuple = (),
    task_kwargs: dict = {},
    sort_output: bool = False,
):
    
    def read_auto(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception:
            pass
        return []
    
    def write_json(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def write_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    input_data = read_auto(input_file)

    dir_path = os.path.dirname(output_file)
    if dir_path: os.makedirs(dir_path, exist_ok=True)
    existing_output = read_auto(output_file)
    if len(input_data) == len(existing_output):
        print('==> 所有任务已完成')
        return

    if key_field is None:
        for i, item in enumerate(input_data):
            item["_auto_key"] = i
        key_field = "_auto_key"

    done_keys = {item[key_field] for item in existing_output}

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    for item in input_data:
        item_key = item[key_field]
        if item_key in done_keys:
            continue

        f = pool.submit(task_fn, item, *task_args, **task_kwargs)
        futures[f] = item_key

    results = existing_output[:]

    if output_file.endswith('jsonl'):
        jsonl_handle = open(output_file, "a", encoding="utf-8")

    for future in tqdm(as_completed(futures), total=len(futures)):
        item_key = futures[future]
        result = future.result()
        if result is None:
            continue
        entry = {key_field: item_key, **result}
        results.append(entry)
        done_keys.add(item_key)

        if output_file.endswith('jsonl'):
            jsonl_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
        elif output_file.endswith('json'):
            write_json(results, output_file)

        if backup_interval > 0 and len(done_keys) % backup_interval == 0:
            base = os.path.dirname(output_file)
            name = os.path.splitext(os.path.basename(output_file))[0]
            if output_file.endswith('jsonl'):
                backup_file = os.path.join(base, f"{name}_{len(done_keys)}.jsonl")
                write_jsonl(results, backup_file)
            if output_file.endswith('json'):
                backup_file = os.path.join(base, f"{name}_{len(done_keys)}.json")
                write_json(results, backup_file)

    pool.shutdown(wait=True)

    if len(input_data) == len(results):
        if sort_output:
            results = sorted(results, key=lambda x: x[key_field])
        if key_field == "_auto_key":
            results = [{k: v for k, v in item.items() if k != "_auto_key"} for item in results]
        if output_file.endswith('jsonl'):
            write_jsonl(results, output_file)
        if output_file.endswith('json'):
            write_json(results, output_file)

if __name__ == "__main__":

    input_file = "./datasets_example/dialogue/result_example.json"
    output_file = "./datasets_example/dialogue/cot_example.json"

    run_with_pool(
        input_file = input_file,
        output_file = output_file,
        task_fn = generate_cot,
        key_field = 'id',
        max_workers = 10,
        sort_output = True,
    )
