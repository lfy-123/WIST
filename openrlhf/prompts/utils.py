import json
from typing import Any, Dict, Optional, List
import re
from collections import OrderedDict

def parse_json_output(raw_text: str) -> Dict[str, Any]:
    """
    Parse the JSON output of the model.
    - Prefer direct json.loads
    - If it fails, try to intercept the substring between the first '{' and the last '}' and parse it
    - If it still fails, throw the original exception so that you can do fallback in the upper layer
    """
    raw_text = raw_text.strip()
    test_output = "="*20 + "Model output start" + "="*20
    test_output += "\n" + raw_text + "\n"
    test_output += "="*20 + "Model output end" + "="*20

    try:
        result = json.loads(raw_text)
        return result, test_output
    except Exception as e:
        pass
    try:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = raw_text[first : last + 1]
            try:
                result = json.loads(candidate)
                return result, test_output
            except Exception as e:
                pass
    except Exception as e:
        print(f"[parse_json_output] Error processing braces: {repr(e)}")
    return None, test_output


def parse_box_output(text: str):
    """
    Parse the content inside \\boxed{...} from the model output.
    - If there are multiple boxed, take the last one
    - If there is no match, return None
    """
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if not matches:
        return None
    ans = matches[-1].strip()
    ans = ans.rstrip(" .")
    return ans if ans else None


def extract_all_boxed_expressions(text):
    """
    Extract the complete content inside all \\boxed{...} and \\boxed{{...}}, including nested curly braces.
    Supports multiple occurrences and nested boxed expressions.
    Returns a list, each item is the complete content inside boxed (not including \\boxed).
    """
    results = []
    pattern = r'\\boxed\{'
    i = 0
    while i < len(text):
        match = re.search(pattern, text[i:])
        if not match:
            break
        start = i + match.start()
        brace_level = 0
        j = start + len(r'\boxed{')
        while j < len(text):
            if text[j] == '{':
                brace_level += 1
            elif text[j] == '}':
                if brace_level == 0:

                    results.append(text[start + len(r'\boxed{'):j])
                    i = j + 1
                    break
                else:
                    brace_level -= 1
            j += 1
        else:

            break
        i = j
    if len(results) == 0:
        return None
    return results[-1]


def extract_mcq_options(text: str) -> Dict[str, str]:
    """
    Extract options from text containing the stem and options.
    Supports:
    - Multi-line format:
        A) xxx
        B) yyy
    - Single-line format:
        ... ? A) xxx; B) yyy; C) zzz; D) www
    Returns:
        OrderedDict, for example {"A": "5 ms", "B": "1 ms", "C": "500 ms", "D": "10 ms"}
    """

    label_pattern = re.compile(r'([A-Za-z])\s*[\)\.\uFF09\u3001\uFF0E]', re.UNICODE)
    matches = list(label_pattern.finditer(text))
    options = OrderedDict()
    if not matches:
        return options
    for idx, m in enumerate(matches):
        label = m.group(1).upper()  
        start = m.end()             
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        chunk = text[start:end].strip()

        chunk = chunk.strip(" \t\n\r;:，。；")
        if chunk:
            options[label] = chunk
    return options

def normalize_pred_answers(preds: List[str], question_text: str) -> List[str]:
    """
    preds: raw output list of the reasoner
    question_text: the text of the question with options, used to extract the option content corresponding to A/B/C/D
    Returns: normalized answer string for each preds[i] (e.g. 'A' or '5 ms')
    """
    options = extract_mcq_options(question_text)
    normalized = []
    for ans in preds:
        if ans is None:
            normalized.append(ans)
            continue
        ans = ans.strip()
        for key, value in options.items():
            if value in ans and key in ans:

                ans = ans.split(value)[0].strip()

                ans = ans.replace(")", "").replace("．", "").replace(".", "").strip()
                break
        normalized.append(ans)
    return normalized

