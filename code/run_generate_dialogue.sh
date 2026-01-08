python generate_dialogue.py \
    --ablation_type full \
    --counselor_model gpt-4.1-mini \
    --seeker_model gpt-4.1-mini \
    --max_workers 16 \

# ablation_type = ['full', 'no_emotion', 'no_safety', 'no_emotion_safety']