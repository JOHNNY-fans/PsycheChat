import os
import time
import openai
import json
from tqdm import tqdm
import regex as re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import *

client = openai.OpenAI(
    api_key='your_api_key',
    base_url='your_base_url'
)

def call_llm(messages, model_name, temperature=0.7, sleep_time=1.0):
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

# Extract JSON object from the model response
def get_json(text):
    pattern = r'\{(?:[^{}]|(?R))*\}'
    try:
        match = re.search(pattern, text)
        if not match:
            return {}
        json_str = match.group(0)
        data = json.loads(json_str)
        return data
    except Exception:
        return {}
    
# Read a JSON file
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = []
    return data

# Write data to a JSON file
def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Role card text
def get_card_text(role_card):
    return (
        f"性别：{role_card['性别']}\n"
        f"年龄：{role_card['年龄']}\n"
        f"职业：{role_card['职业']}\n"
        f"人格：\n"
        f"  开放性：{role_card['人格']['开放性']}\n"
        f"  尽责性：{role_card['人格']['尽责性']}\n"
        f"  外向性：{role_card['人格']['外向性']}\n"
        f"  宜人性：{role_card['人格']['宜人性']}\n"
        f"  神经质：{role_card['人格']['神经质']}\n"
        f"性格：{role_card['性格']}\n"
        f"语言风格：{role_card['语言风格']}\n"
        f"爱好：{role_card['爱好']}\n"
        f"问题：{role_card['问题']}\n"
        f"目标：{role_card['目标']}\n"
        f"内心独白：{role_card['内心独白']}\n"
        f"初始情绪：{role_card['初始情绪']}\n"
        f"事件触发情绪：{role_card['事件触发情绪']}"
    )

# Emotion memory text
def get_memory_text(emotion_memory):
    return (
        "<emotion_output>\n"
        f"当前情绪：{'，'.join(emotion_memory['current_emotion'])}\n"
        f"当前情绪原因：{emotion_memory['current_analysis']}\n"
        f"近期转变：{emotion_memory['recent_change'] or '无'}\n"
        f"整体趋势：{emotion_memory['overall_trend'] or '无'}\n"
        f"情绪转变原因：{emotion_memory['shift_analysis'] or '无'}\n"
        "</emotion_output>"
    )

# Safety analysis text
def get_safety_text(counselor_response, seeker_utterances, safety_res):
    output = ["<safety_output>"]
    output.append(f'如果咨询师回复：{counselor_response}')
    output.append('来访者可能的反应：')
    safety_analysis = safety_res.get('safety_analysis', [])
    for item in safety_analysis:
        uid = item['uid']
        analysis = item['analysis']
        is_safe = item['pass']
        utterance = next((u['utterance'] for u in seeker_utterances if u['uid'] == uid), '')
        output.append(f'回复：{utterance}')
        if is_safe:
            output[-1] += '（不存在风险）'
        else:
            output[-1] += '（存在风险）'
        output.append(f'分析：{analysis}\n')
    suggestion_emotion = safety_res['suggestion'].get('emotion', '')
    suggestion_safety = safety_res['suggestion'].get('safety', '')
    output.append(f'情绪修改建议：{suggestion_emotion}')
    output.append(f'安全修改建议：{suggestion_safety}')
    output.append("</safety_output>")
    return '\n'.join(output)

# Dialogue history text
def get_history_text(history):
    return '\n'.join([f"{h['role']}：{h['content']}" for h in history])

# Emotion validation
emotion_groups = [
    ["平静", "快乐", "狂喜"],
    ["接受", "信任", "崇敬"],
    ["担心", "恐惧", "惊悚"],
    ["不解", "惊讶", "惊诧"],
    ["伤感", "悲伤", "悲痛"],
    ["厌倦", "厌恶", "憎恨"],
    ["烦躁", "生气", "暴怒"],
    ["关心", "期待", "警惕"]
]
emotion_to_group = {emo: idx for idx, group in enumerate(emotion_groups) for emo in group}

def validate_emotion(emotion_list):
    if not isinstance(emotion_list, list):
        return False
    if not (1 <= len(emotion_list) <= 3):
        return False
    used_groups = set()
    for emo in emotion_list:
        if not isinstance(emo, str):
            return False
        if emo not in emotion_to_group:
            return False
        group = emotion_to_group[emo]
        if group in used_groups:
            return False
        used_groups.add(group)
    return True

# Generate a seeker response
def seeker_generate(result, messages_seeker, model_name, max_llm_retries):
    try_num = 0
    while True:
        try_num += 1
        if try_num > max_llm_retries:
            raise RuntimeError
        try:
            seeker_res = call_llm(messages=messages_seeker, model_name=model_name)
            seeker_res = get_json(seeker_res)
            seeker_emotion = seeker_res['emotion']
            if not validate_emotion(seeker_emotion):
                continue
            seeker_event = seeker_res['event']
            seeker_goals = seeker_res['goals']
            seeker_response = seeker_res['response']
            if show_log: print(seeker_res)
            break
        except Exception:
            continue
    result.append({'module': 'seeker', **seeker_res})
    end_flag = False
    if 'END' in seeker_response:
        seeker_response = seeker_response.replace('END', '').strip()
        if seeker_response.strip() != '':
            end_flag = True
    return result, seeker_res, seeker_response, end_flag

# Generate and validate emotion analysis
def emotion_generate(result, messages_emotion, turn_num, model_name, max_llm_retries):
    try_num = 0
    while True:
        try_num += 1
        if try_num > max_llm_retries:
            raise RuntimeError
        try:
            emotion_res = call_llm(messages=messages_emotion, model_name=model_name)
            emotion_res = get_json(emotion_res)
            emotion_thinking = emotion_res['thinking']
            current_emotion = emotion_res['current_emotion']
            if not validate_emotion(current_emotion):
                continue
            current_analysis = emotion_res['current_analysis']
            recent_change = emotion_res['recent_change']
            overall_trend = emotion_res['overall_trend']
            shift_analysis = emotion_res['shift_analysis']
            if turn_num <= 3 and current_analysis == '':
                continue
            if turn_num > 3 and '' in (current_analysis, recent_change, overall_trend, shift_analysis):
                continue
            if show_log: print(emotion_res)
            break
        except Exception:
            continue
    result.append({'module': 'emotion', **emotion_res})
    emotion_memory = emotion_res
    return result, emotion_memory

# Generate counselor response
def counselor_generate(result, messages_counselor, model_name, max_llm_retries):
    try_num = 0
    while True:
        try_num += 1
        if try_num > max_llm_retries:
            raise RuntimeError
        try:
            counselor_res = call_llm(messages=messages_counselor, model_name=model_name)
            counselor_res = get_json(counselor_res)
            counselor_thinking = counselor_res['thinking']
            counselor_phase = counselor_res['phase']
            counselor_strategy = counselor_res['strategy']
            counselor_response = counselor_res['response']
            if '' in (counselor_phase, counselor_strategy, counselor_response):
                continue
            if show_log: print(counselor_res)
            break
        except Exception:
            continue
    result.append({'module': 'counselor', **counselor_res})
    return result, counselor_res, counselor_response

# Simulate possible seeker reactions to counselor response
def simulate_generate(result, messages_simulate, model_name, max_llm_retries):
    try_num = 0
    while True:
        try_num += 1
        if try_num > max_llm_retries:
            raise RuntimeError
        try:
            simulate_res = call_llm(messages=messages_simulate, model_name=model_name)
            pattern = r"\[\s*{[\s\S]*?}\s*\]"
            match = re.search(pattern, simulate_res)
            if not match:
                continue
            simulate_res = match.group(0).strip()
            simulate_res = json.loads(simulate_res)
            for item in simulate_res:
                simulate_type = item['type']
                simulate_response = item['response']
            if show_log: print(simulate_res)
            break
        except Exception:
            continue
    seeker_utterances = [{'uid': i+1, 'utterance': item['response']} for i, item in enumerate(simulate_res)]
    result.append({'module': 'simulate', 'utterances': simulate_res})
    return result, seeker_utterances

# Perform safety evaluation on counselor response
def safety_generate(result, messages_safety, model_name, max_llm_retries):
    try_num = 0
    while True:
        try_num += 1
        if try_num > max_llm_retries:
            raise RuntimeError
        try:
            safety_res = call_llm(messages=messages_safety, model_name=model_name)
            safety_res = get_json(safety_res)
            safety_thinking = safety_res['thinking']
            safety_analysis = safety_res['safety_analysis']
            any_false = False
            for item in safety_analysis:
                uid = item['uid']
                analysis = item['analysis']
                is_safe = item['pass']
                if not is_safe:
                    any_false = True
            pass_flag = safety_res['pass_flag']
            suggestion = safety_res['suggestion']
            suggestion_emotion = safety_res['suggestion']['emotion']
            suggestion_safety = safety_res['suggestion']['safety']
            if any_false and pass_flag:
                continue
            if pass_flag == False and suggestion_emotion == '' and suggestion_safety == '':
                continue
            if show_log: print(safety_res)
            break
        except Exception:
            continue
    result.append({'module': 'safety', **safety_res})
    return result, safety_res, pass_flag

prompt_counselor_dict =  {
    'full': prompt_counselor_full,
    'no_emotion': prompt_counselor_no_emotion,
    'no_safety': prompt_counselor_no_safety,
    'no_emotion_safety': prompt_counselor_no_emotion_safety
}

# Generate a counseling dialogue
def generate(card, ablation_type, counselor_model, seeker_model, show_log=False, show_utterance=False, max_global_retries=3):
    
    global_try_num = 0
    while True:

        global_try_num += 1
        if global_try_num > max_global_retries:
            return {'topic': topic, 'history': [], 'result': []}
        
        try:
            history_seeker = []
            history_counselor = []
            history = []
            result = []
            emotion_memory = {}
            max_llm_retries = 30
            max_safety_retries = 5
            max_turns = 30

            topic = card['topic']
            role_card = card['role_card']
            result.append({'module': 'card', 'role_card': role_card})
            role_card = get_card_text(role_card)
            if show_log: print(role_card)

            system_seeker = prompt_seeker.replace('{{role_card}}', role_card)
            system_counselor = prompt_counselor_dict[ablation_type]
        
            turn_num = 0
            while True:

                turn_num += 1
                if turn_num > max_turns:
                    raise RuntimeError
                
                # Seeker response
                if history_seeker == []:
                    history_seeker.append({'role': 'user', 'content': '请你以一个简短的回复开启咨询'})
                messages_seeker = [{'role': 'system', 'content': system_seeker}] + history_seeker
                result, seeker_res, seeker_response, end_flag = seeker_generate(result, messages_seeker, counselor_model, max_llm_retries)
                if show_utterance: print(f'来访者：{seeker_response}')
                history_seeker.append({'role': 'assistant', 'content': json.dumps(seeker_res, indent=4, ensure_ascii=False)})
                history_counselor.append({'role': 'user', 'content': seeker_response})
                history.append({'role': '来访者', 'content': seeker_response})
                if end_flag:
                    return {'topic': topic, 'history': history, 'result': result}

                # Emotion analysis
                if ablation_type in ['full', 'no_safety']:
                    user_emotion = prompt_emotion_analysis.replace('{{history}}', json.dumps(history[:-1], indent=4, ensure_ascii=False)).replace('{{seeker_utterance}}', history[-1]['content']).replace('{{emotion_memory}}', json.dumps(emotion_memory, indent=4, ensure_ascii=False))
                    messages_emotion = [{'role': 'user', 'content': user_emotion}]
                    result, emotion_memory = emotion_generate(result, messages_emotion, turn_num, counselor_model, max_llm_retries)
                    emotion_text = get_memory_text(emotion_memory)

                if ablation_type in ['full', 'no_emotion']:
                    pass_flag = False
                    safety_analysis = []
                    safety_num = 0
                    while not pass_flag:
                        safety_num += 1
                        if safety_num > max_safety_retries:
                            raise RuntimeError

                        # Counselor response
                        messages_counselor = [{'role': 'system', 'content': system_counselor}] + history_counselor
                        if ablation_type == 'full':
                            messages_counselor = messages_counselor + [{'role': 'assistant', 'content': emotion_text}]
                        if safety_analysis != []:
                            safety_text = get_safety_text(counselor_response, seeker_utterances, safety_res)
                            messages_counselor = messages_counselor + [{'role': 'assistant', 'content': safety_text}]
                        result, counselor_res, counselor_response = counselor_generate(result, messages_counselor, counselor_model, max_llm_retries)

                        # Safety analysis
                        user_simulate = prompt_simulate.replace('{{history}}', json.dumps(history, indent=4, ensure_ascii=False)).replace('{{counselor_utterance}}', counselor_response)
                        messages_simulate = [{'role': 'user', 'content': user_simulate}]
                        result, seeker_utterances = simulate_generate(result, messages_simulate, seeker_model, max_llm_retries)

                        user_safety = prompt_safety_analysis.replace('{{history}}', json.dumps(history, indent=4, ensure_ascii=False)).replace('{{counselor_utterance}}', counselor_response).replace('{{seeker_utterances}}', json.dumps(seeker_utterances, indent=4, ensure_ascii=False))
                        messages_safety = [{'role': 'user', 'content': user_safety}]
                        result, safety_res, pass_flag = safety_generate(result, messages_safety, counselor_model, max_llm_retries)

                elif ablation_type in ['no_safety', 'no_emotion_safety']:
                    messages_counselor = [{'role': 'system', 'content': system_counselor}] + history_counselor
                    if ablation_type == 'no_safety':
                        messages_counselor = messages_counselor + [{'role': 'assistant', 'content': emotion_text}]
                    result, counselor_res, counselor_response = counselor_generate(result, messages_counselor, counselor_model, max_llm_retries)

                if show_utterance: print(f'咨询师：{counselor_response}')
                history_counselor.append({'role': 'assistant', 'content': json.dumps(counselor_res, indent=4, ensure_ascii=False)})
                history_seeker.append({'role': 'user', 'content': counselor_response})
                history.append({'role': '咨询师', 'content': counselor_response})

        except RuntimeError:
            continue
        except KeyboardInterrupt:
            return {'topic': topic, 'history': [], 'result': []}
        except Exception:
            return {'topic': topic, 'history': [], 'result': []}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_type', default='full', type=str)
    parser.add_argument('--counselor_model', default='gpt-4.1-mini', type=str)
    parser.add_argument('--seeker_model', default='gpt-4.1-mini', type=str)
    parser.add_argument('--show_log', action='store_true')
    parser.add_argument('--show_utterance', action='store_true')
    parser.add_argument('--max_workers', default=8, type=int)
    parser.add_argument('--generate_num', default=-1, type=int)
    args = parser.parse_args()

    ablation_type = args.ablation_type
    counselor_model = args.counselor_model
    seeker_model = args.seeker_model
    show_log = args.show_log
    show_utterance = args.show_utterance
    max_workers = args.max_workers
    generate_num = args.generate_num

    card_path = './datasets_example/role_card/role_cards.json'
    history_path = f'./datasets_example/dialogue/history_example.json'
    result_path = f'./datasets_example/dialogue/result_example.json'

    cards = read_json(card_path)
    if generate_num > 0: cards = cards[:generate_num]
    history_json = read_json(history_path)
    result_json = read_json(result_path)
    done_history = {item["id"]: {"topic": item["topic"], "history": item["history"]} for item in history_json}
    done_result = {item["id"]: {"topic": item["topic"], "result": item["result"]} for item in result_json}

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    for card in cards:
        idx = card["id"]
        if idx in done_history:
            continue

        future = pool.submit(generate, card, ablation_type, counselor_model, seeker_model, show_log, show_utterance)
        futures[future] = idx

    for future in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[future]
        result = future.result()
        topic = result.get("topic")
        history = result.get("history")
        res = result.get("result")
        if not history or not res:
            continue
        done_history[idx] = {
            "topic": topic,
            "history": history
        }
        done_result[idx] = {
            "topic": topic,
            "result": res
        }
        history_json = [
            {"id": k, "topic": v["topic"], "history": v["history"]}
            for k, v in sorted(done_history.items(), key=lambda x: x[0])
        ]
        result_json = [
            {"id": k, "topic": v["topic"], "result": v["result"]}
            for k, v in sorted(done_result.items(), key=lambda x: x[0])
        ]
        write_json(history_json, history_path)
        write_json(result_json, result_path)
        if len(history_json) % 100 == 0:
            base_dir = os.path.dirname(history_path)
            history_filename = os.path.splitext(os.path.basename(history_path))[0]
            result_filename = os.path.splitext(os.path.basename(result_path))[0]
            history_backup_name = f"{history_filename}_{len(history_json)}.json"
            result_backup_name = f"{result_filename}_{len(result_json)}.json"
            history_backup_file = os.path.join(base_dir, history_backup_name)
            result_backup_file = os.path.join(base_dir, result_backup_name)
            write_json(history_json, history_backup_file)
            write_json(result_json, result_backup_file)

    pool.shutdown(wait=True)