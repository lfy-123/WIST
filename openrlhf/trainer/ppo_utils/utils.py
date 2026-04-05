import re
import unicodedata

MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "exp", "sqrt", "pi", "π", "mod"
}

EXPLANATION_KEYWORDS = [
    "to solve", "we first", "first", "second", "third",
    "step", "steps", "next", "then", "after that",
    "therefore", "thus", "hence",
    "in conclusion", "this means",
    "by allowing", "by using", "in order to",
    "show your steps", "show your work", "explain",
]

PLACEHOLDER_ANSWER_PATTERNS = [
    "your answer",
    "the answer here",
    "answer derivation",
    "your derived answer",
    "answer details",
    "...",
    "derived answer",
    "your derived answer",
    "the answer here",
    "answer",
    "solution"
]


def _looks_like_placeholder(text: str, patterns) -> bool:
    lower = text.strip().lower()
    return any(p in lower for p in patterns)


def _is_pure_integer(ans: str) -> bool:
    return re.fullmatch(r"[+-]?\d+", ans.strip()) is not None


def _is_pure_real_number(ans: str) -> bool:
    return re.fullmatch(
        r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?",
        ans.strip()
    ) is not None


MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "exp", "sqrt", "pi",
    "frac", "sum", "prod", "int", "lim",
    "min", "max", "arg", "sup", "inf",
    "cdot", "mod",
}


MATH_OPS = set("+-*/^=<>")


def _strip_latex_math_wrappers(s: str) -> str:
    """
    去掉常见的 LaTeX 数学环境外壳：
    \( ... \), \[ ... \], $...$, $$...$$
    只保留里面的内容做表达式判断。
    """
    s = s.strip()
# TODO: translate comment
    wrappers = [
        ("$$", "$$"),
        ("\\[", "\\]"),
        ("\\(", "\\)"),
        ("$", "$"),
    ]
    for left, right in wrappers:
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            return s[len(left):-len(right)].strip()
    return s

def _is_pure_expression(ans: str) -> bool:
    """
    更宽松、支持 LaTeX/Unicode 的表达式判断：
    - 先剥掉 \(...\)、\[...\]、$...$ 等外壳；
    - 允许任意 Unicode 字母（含希腊字母）+ 数字 + 常见数学符号；
    - 不允许换行；
    - 必须包含至少一个运算符/比较符 (+-*/^=<>|) 或至少一个 LaTeX 命令；
    - 过滤掉“英文长单词太多”的自然语言句子。
    """
    if not isinstance(ans, str):
        return False

    s = _strip_latex_math_wrappers(ans).strip()
    if not s:
        return False

# 1) TODO: translate comment
    if "\n" in s:
        return False

# 2) TODO: translate comment
    has_op = any(c in MATH_OPS for c in s)

# 3) TODO: translate comment
    latex_cmd_matches = list(re.finditer(r"\\([A-Za-z]+)", s))
    has_latex_cmd = len(latex_cmd_matches) > 0

# TODO: translate comment
    if not (has_op or has_latex_cmd):
        return False

# 4) TODO: translate comment
    banned_punct = set("?!？！」」")
    if any(c in banned_punct for c in s):
        return False

# 5) TODO: translate comment
# TODO: translate comment
# - TODO: translate comment
# - TODO: translate comment
# - TODO: translate comment
# - TODO: translate comment
# - TODO: translate comment
# - TODO: translate comment
    allowed_punct = set("+-*/^=.,()[]{}'_|<>\\&%:;@$`\"")

    for ch in s:
        if ch.isspace():
            continue
        if ch.isdigit():
            continue
        if ch in allowed_punct:
            continue

        cat = unicodedata.category(ch)
# TODO: translate comment
        if cat.startswith("L"):
            continue
# TODO: translate comment
        if cat.startswith("M"):
            continue
# TODO: translate comment
        if cat.startswith("S"):
            continue

# TODO: translate comment
        return False

# 6) TODO: translate comment
# TODO: translate comment
# TODO: translate comment
    all_words = re.findall(r"[A-Za-z]{3,}", s)
    cmd_words = {m.group(1).lower() for m in latex_cmd_matches}
    math_words = set(MATH_WORDS) | cmd_words

    non_math_words = [w.lower() for w in all_words if w.lower() not in math_words]
# TODO: translate comment
    if len(non_math_words) > 2:
        return False

    return True




def infer_answer_type_from_value(answer: str) -> str:
    """
    不信任模型给出的 answer_type，
    只根据 answer 自身形式推断类型：
    - 纯整数 → integer
    - 纯实数 → real_number
    - 纯数学表达式 → expression
    - 其他（任何“表述 + 数值/表达式”） → string
    """
    # if not isinstance(answer, str):
    #     return "string"
    ans = answer.strip()
    # if not ans:
    #     return "string"
    if _is_pure_integer(ans):
        return "integer"
    if _is_pure_real_number(ans):
        return "real_number"
    if _is_pure_expression(ans):
        return "expression"
    return "string"



def is_valid_string_answer_shape(ans: str,
                                 max_tokens: int = 5) -> bool:
    """
    严格版本的 string 答案形式校验：
    - 非空、单行；
    - 长度和 token 数不超阈值；
    - 不能是占位符或“Answer: …”这类模板；
    - 不应该看起来像长篇解释（多句、含明显解题步骤关键词）。
    """
    if not isinstance(ans, str):
        return False
    s = ans.strip()
    if not s:
        return False
# TODO: translate comment
    if _looks_like_placeholder(s, PLACEHOLDER_ANSWER_PATTERNS):
        return False
    lower = s.lower()
# TODO: translate comment
    if "answer" in lower:
        return False
# TODO: translate comment
    if "\n" in s:
        return False
# # TODO: translate comment
    # if len(s) > max_answer_length:
    #     return False
    tokens = re.findall(r"\S+", s)
    if len(tokens) > max_tokens:
        return False
# TODO: translate comment
    sentence_end_count = sum(ch in ".?!。？!" for ch in s)
    if sentence_end_count > 1:
        return False
# TODO: translate comment
    if any(k in lower for k in EXPLANATION_KEYWORDS):
        return False
    return True

def has_abcd_options(text: str) -> bool:
    """
    检查文本中是否出现类似 'A) xxx' / 'B. yyy' / 'C) zzz' / 'D. www' 的四个选项。
    只用来排除把自由题生成成选择题的情况。
    """
    if not isinstance(text, str) or not text.strip():
        return False
    pattern = re.compile(r'(?:^|\s)([A-D])[\.\)]\s+', re.MULTILINE)
    labels = {m.group(1) for m in pattern.finditer(text)}
    return all(ch in labels for ch in "ABCD")

def is_valid_free_qa_from_model_output(question: str,
                                       answer: str,
                                       min_question_len: int = 50,
                                       max_answer_tokens: int = 5) -> bool:
    """
    FREE 题型输出校验（不信任模型给的 answer_type）：
    1. 问题：
       - 非空；
       - 不是选择题；
       - 去掉前缀 Question/Q 和标点后长度 ≥ min_question_len。
    2. 答案：
       - 先根据内容推断类型 ∈ {"integer","real_number","expression","string"}；
       - 若推断为 integer / real_number / expression：
           只要形式匹配对应类型且长度不过长，就认为形式合法；
       - 若推断为 string：
           用更严格的 string 规则过滤掉占位符、解释性长句等。
    """

# ---------- TODO: translate comment

    if not isinstance(question, str) or not question.strip():
        return False

# TODO: translate comment
    if has_abcd_options(question):
        return False

# TODO: translate comment
    stem = re.sub(r'^\s*(Question|Q)\s*:\s*', '', question,
                  flags=re.IGNORECASE).strip()
    stem_core = re.sub(r'[\.\?？!！…\s]', '', stem)  # TODO: translate comment

    if len(stem_core) < min_question_len:
        return False

# ---------- TODO: translate comment

    if not isinstance(answer, str):
        return False
    ans = answer.strip()
    if not ans:
        return False

    if "undefined" in ans:
        return False

# TODO: translate comment
    inferred_type = infer_answer_type_from_value(ans)

# TODO: translate comment
    if inferred_type in {"integer", "real_number", "expression"}:
# TODO: translate comment
        if not is_valid_answer_by_type(ans, inferred_type):
            return False
        # if len(ans.split()) > max_answer_length:
        #     return False
        return True

# TODO: translate comment
    if not is_valid_string_answer_shape(ans, max_tokens=max_answer_tokens):
        return False

# string TODO: translate comment
    return True

def is_valid_answer_by_type(answer: str, answer_type: str) -> bool:
    """
    根据 answer_type 检查 correct_answer 的基本形式。
    这里只做“形式”检查，不做数学正确性验证。
    """
    if not isinstance(answer, str):
        return False
    ans = answer.strip()
    if not ans:
        return False
    if answer_type == "integer":
        return _is_pure_integer(ans)
    if answer_type == "real_number":
        return _is_pure_real_number(ans)
    if answer_type == "expression":
        return _is_pure_expression(ans)
    if answer_type == "string":
# TODO: translate comment
# TODO: translate comment
        core = re.sub(r'[.\·•…\s]', '', ans)
        return len(core) >= 1
    return False





def preprocess_data(
    question: str = "",
    apply_chat_template = None,
    template_type: str | None = None,  # TODO: translate comment
):
    """
    data: 一条样本，如 {"input": "...", "label": "..."} 之类
    input_key: data 中问题字段名，比如 "input" / "question"
    label_key: data 中标签字段名，比如 "label" / "answer"，为 None 时返回空串
    apply_chat_template: 一般是 tokenizer.apply_chat_template
    template_type: 
        - "qwen3"        -> 使用 Qwen3 Family 模板
        - "octothinker"  -> 使用 OctoThinker Family 模板
        - None           -> 走原来的逻辑 (input_template / apply_chat_template)
    """
# 1) Qwen3 Family TODO: translate comment
    if template_type == "qwen3":
# TODO: translate comment
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
# TODO: translate comment
            prompt = (
                "<|im_start|>system You are a helpful assistant. <|im_end|> "
                f"<|im_start|>user {question} \nPlease reason step by step, "
                "and put your final answer within \\boxed{}. \n<|im_end|> "
                "<|im_start|>assistant "
            )
# 2) OctoThinker Family TODO: translate comment
    elif template_type == "octothinker":
        prompt = (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer.\n\n"
            f"User: {question}\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            "Assistant:"
        )
    else:
        raise ValueError(f"Unknown template_type: {template_type}")
    return prompt



def preprocess_challenger_data(
    question: str = "",
    apply_chat_template = None,
    template_type: str | None = None,  # TODO: translate comment
):
    """
    data: 一条样本，如 {"input": "...", "label": "..."} 之类
    input_key: data 中问题字段名，比如 "input" / "question"
    label_key: data 中标签字段名，比如 "label" / "answer"，为 None 时返回空串
    apply_chat_template: 一般是 tokenizer.apply_chat_template
    template_type: 
        - "qwen3"        -> 使用 Qwen3 Family 模板
        - "octothinker"  -> 使用 OctoThinker Family 模板
        - None           -> 走原来的逻辑 (input_template / apply_chat_template)
    """
# 1) Qwen3 Family TODO: translate comment
    if template_type == "qwen3":
# TODO: translate comment
        if apply_chat_template is not None:
            chat = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": question
                },
            ]
            prompt = apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
# TODO: translate comment
            prompt = (
                "<|im_start|>system You are a helpful assistant. <|im_end|> "
                f"<|im_start|>user {question}\n<|im_end|> "
                "<|im_start|>assistant "
            )
# 2) OctoThinker Family TODO: translate comment
    elif template_type == "octothinker":
        prompt = (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer.\n\n"
            f"User: {question}\n"
            "Assistant:"
        )
    else:
        raise ValueError(f"Unknown template_type: {template_type}")
    return prompt



if __name__ == "__main__":
    question = ""
    answer = 'x >= 3500'
    # answer = "2, 3, 5"
    answer_type = infer_answer_type_from_value(answer)
    print(answer_type)
    
    print(is_valid_answer_by_type(answer, answer_type))


