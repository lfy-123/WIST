
def preprocess_data(
    question: str = "",
    apply_chat_template = None,
    template_type: str | None = None,
):
    """
    data: A sample, such as {"input": "...", "label": "..."}
    input_key: Question field name in data, e.g., "input" / "question"
    label_key: Label field name in data, e.g., "label" / "answer", returns empty string if None
    apply_chat_template: Typically tokenizer.apply_chat_template
    template_type: 
        - "qwen3"        -> Use Qwen3 Family template
        - "octothinker"  -> Use OctoThinker Family template
        - None           -> Use original logic (input_template / apply_chat_template)
    """
    # 1) Qwen3 Family
    if template_type == "qwen3":
        if apply_chat_template is not None:
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        f"{question} \nPlease reason step by step, "
                        "and put your final answer within \\boxed{}."
                    ),
                },
            ]
            prompt = apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "<|im_start|>system You are a helpful assistant. <|im_end|> "
                f"<|im_start|>user {question} \nPlease reason step by step, "
                "and put your final answer within \\boxed{}. \n<|im_end|> "
                "<|im_start|>assistant "
            )
    # 2) OctoThinker Family
    elif template_type == "octothinker":
        prompt = (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer.\n\n"
            f"User: {question}\n"
            "You must put your answer inside \\boxed{}.\n\n"
            "Assistant:"
        )
    else:
        raise ValueError(f"Unknown template_type: {template_type}")
    return prompt


