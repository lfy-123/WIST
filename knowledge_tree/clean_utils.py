from typing import List, Dict
import requests
import re
from typing import List, Optional, Set
import json
from resiliparse.extract.html2text import extract_plain_text as resiliparse_extract_plain_text
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher

DISPLAY_START = "{\\displaystyle"

class HTMLExtractorAlgorithm(ABC):
    NON_SPACED_LANGUAGES = frozenset(["THAI", "CHINESE", "JAPANESE", "KOREAN"])

    @abstractmethod
    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        pass

class ResiliparseExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        required_stopword_density: float = 0.32,
        main_content: bool = True,
        alt_texts: bool = False,
    ):
        """
        Initialize the Resiliparse text extraction algorithm with specified parameters.

        The Resiliparse algorithm extracts structural or semantic information from noisy raw web data for further processing,
        such as (main) content extraction / boilerplate removal, schema extraction, general web data cleansing, and more.

        It is implemented via the `extract_plain_text` function in the `resiliparse.extract.html2text` module.
        Resiliparse HTML2Text is a very fast and rule-based plain text extractor for HTML pages which uses the Resiliparse DOM parser.
        The `extract_plain_text` function extracts all visible text nodes inside the HTML document's <body>.
        Only <script>, <style> and a few other (generally) invisible elements are skipped and very basic ASCII formatting is applied.

        Please refer to the Resiliparse documentation for more details: https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html

        NeMo Curator has added a stopword density filter to the Resiliparse extraction process, which requires that a paragraph contains a certain proportion of stopwords.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            main_content: Whether to apply simple heuristics for extracting only "main-content" elements.
            alt_texts: Whether to preserve alternative text descriptions (e.g., for images).

        """
        self.required_stopword_density = required_stopword_density
        self.main_content = main_content
        self.alt_texts = alt_texts

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        text = resiliparse_extract_plain_text(html, main_content=self.main_content, alt_texts=self.alt_texts)
        paragraphs = list(filter(None, text.split("\n")))

        if language in self.NON_SPACED_LANGUAGES:
            print("stopword_density is ignored for non-space-separated languages.")
            result = paragraphs
        else:
            result = []
            for paragraph in paragraphs:
                words = paragraph.split()
                length = len(words)
                if length == 0:
                    continue
                stopwords = [word for word in words if word in stop_words]
                stopword_density = len(stopwords) / length
                if stopword_density >= self.required_stopword_density:
                    result.append(paragraph)
        return result



def clean_paragraphs(
    paragraphs: List[str],
    stopwords: Optional[Set[str]] = None,
    keep_formula_window: int = 3,
    min_sentence_len: int = 20,
    url = "",
) -> List[str]:
    """
    Perform a secondary cleaning on the full paragraphs extracted by the extractor:
    1. Find the "first truly text-like sentence" as the starting position.
    2. Find the "last text-like sentence" as the ending position.
    3. Crop head and tail.
    4. Look back a few paragraphs to keep potential formulas together.
    5. Delete obvious navigation/footer boilerplate.
    
    Args:
        paragraphs: List[str]. List of paragraphs obtained from Resiliparse (or other extractor), in original order.
        stopwords: Optional[Set[str]]. Stopword set used to assist in determining "whether it looks like a natural language sentence".
        keep_formula_window: int. Window size (number of paragraphs) to look back from the main text starting point to preserve suspected formula paragraphs.
        min_sentence_len: int. Minimum number of characters to be considered "sentence-like", too short is usually menus/titles.

    Returns:
        List[str]. Cleaned list of paragraphs (still keeping original order).
    """
    SENT_END = r"[.!?。？！]"  
    STRONG_BOILERPLATE_PATTERNS = [
        "privacy policy",
        "cookie",
        "terms of use",
        "terms of service",
        "by using this site",
        "all rights reserved",
        "copyright",
        "log in",
        "sign in",
        "create account",
        "subscribe",
        "home page",
        "navigation",
        "main menu",
        "edit this page",
        "view history",
        "download as pdf",
        "mobile view",
        "contact us",
        "developers",
    ]
    WEAK_BOILERPLATE_PATTERNS = [
        "statistics",   
        "edit",
        "edited",
    ]
    ERROR_PATTERNS = [
        "404",
        "not found",
        "forbidden",
        "access denied",
        "error",
        "bad gateway",
    ]
    FORMULA_TOKENS = [
        "$", "\\(", "\\)", "\\[", "\\]",
        "\\frac", "\\sum", "\\int", "\\lim", "\\prod",
        "_{", "^{", "\\alpha", "\\beta", "\\gamma",
    ]
    def normalize(text: str) -> str:

        return (
            text.replace("\xa0", " ")
                .replace("\u200d", "")  # zero width joiner
                .strip()
        )
    def is_obvious_boilerplate(text: str) -> bool:
        low = text.lower().strip()

        if any(pat in low for pat in STRONG_BOILERPLATE_PATTERNS):
            return True

        if len(low) <= 40 and any(pat in low for pat in WEAK_BOILERPLATE_PATTERNS):
            return True
        return False
    def is_error_page(text: str) -> bool:
        low = text.lower()
        return any(pat in low for pat in ERROR_PATTERNS)
    def looks_like_formula(text: str) -> bool:
        return any(tok in text for tok in FORMULA_TOKENS)
    def stopword_ratio(text: str) -> float:
        if not stopwords:
            return 0.0
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0
        sw_count = sum(tok in stopwords for tok in tokens)
        return sw_count / len(tokens)
    def is_sentence_like(text: str) -> bool:
        t = text.strip()
        if len(t) < min_sentence_len:
            return False
        if text[-1] not in [".", "!", "?", "。", "！", "？", "...", ":", ";", "-", ")"]:
            return False
        letters = sum(ch.isalpha() for ch in t)
        if letters / max(len(t), 1) < 0.4:
            return False
        if stopwords:
            if stopword_ratio(t) < 0.05:
                return False
        return True
    def find_first_content_idx(paras: List[str]) -> int:
        for i, p in enumerate(paras):
            if not p:
                continue
            if is_obvious_boilerplate(p):
                continue
            if is_error_page(p):

                continue
            if is_sentence_like(p):
                return i
        return 0
    def find_last_content_idx(paras: List[str], start: int) -> int:
        last = len(paras) - 1
        for i in range(last, start - 1, -1):
            p = paras[i]
            if not p:
                continue
            if is_obvious_boilerplate(p):
                continue
            if is_sentence_like(p):
                return i
        return last

    if not paragraphs:
        return []
    norm_paras = [normalize(p) for p in paragraphs]
    pre_start_idx = 0
    if "wikihow.com" in url:
        check_content = "wikiHow is where trusted research and expert knowledge come together. Learn why people trust wikiHow".lower()
        for idx, p in enumerate(norm_paras):
            if check_content in p.lower():
                pre_start_idx = idx + 1
                break
        norm_paras = norm_paras[pre_start_idx:]
    if "quantamagazine.org" in url:
        check_content = "Create a reading list by clicking the Read Later icon next to the articles you wish to save".lower()
        for idx, p in enumerate(norm_paras):
            if check_content in p.lower():
                pre_start_idx = idx + 1
                break
        norm_paras = norm_paras[pre_start_idx:]
    if "chapter" in url:
        check_content = "Want to create or adapt books like this? Learn more about how Pressbooks supports open publishing practices".lower()
        for idx, p in enumerate(norm_paras):
            if check_content in p.lower():
                pre_start_idx = idx + 1
                break
        norm_paras = norm_paras[pre_start_idx:]
    if "brilliant.org/wiki" in url:
        check_content = "Reset password New user? Sign up".lower()
        for idx, p in enumerate(norm_paras):
            if check_content in p.lower():
                pre_start_idx = idx + 1
                break
        norm_paras = norm_paras[pre_start_idx:]

    start_idx = find_first_content_idx(norm_paras)
    end_idx = find_last_content_idx(norm_paras, start_idx)
    if start_idx > end_idx:
        return []
    if keep_formula_window > 0:
        for i in range(max(0, start_idx - keep_formula_window), start_idx):
            p = norm_paras[i]
            if not p:
                continue
            if is_obvious_boilerplate(p):
                continue
            if looks_like_formula(p):
                start_idx = i
                break
    sliced = norm_paras[start_idx : end_idx + 1]
    cleaned = []
    for p in sliced:
        if not p:
            continue
        if is_obvious_boilerplate(p):
            continue
        cleaned.append(p)
    return cleaned



def extract_text_from_html(html: str) -> str:
    """
    Extract plain text from HTML, removing irrelevant content like script/style.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned

def fetch_page_text(url: str, timeout: int = 10) -> str:
    """
    Request the webpage and return the extracted text.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, allow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding  
    return resp.text


def clean_line(line: str) -> str:
    def find_displaystyle_spans(s: str):
        """
        Find the start, end, and internal contents of all {\\displaystyle ...} (supports nested braces)
        Returns: [(start, end, inside), ...]
        """
        spans = []
        key = '{\\displaystyle'
        n = len(s)
        i = 0
        while True:
            idx = s.find(key, i)
            if idx == -1:
                break
            depth = 1  
            j = idx + len(key)
            while j < n and depth > 0:
                ch = s[j]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                end = j
                inside = s[idx + len(key): j-1]  
                spans.append((idx, end, inside))
                i = end
            else:
                break
        return spans

    def token_type(tok: str) -> str:
        """Categorize into three simple classes: natural language word / punctuation / math"""
        if tok.isalpha() and len(tok) >= 2:
            return "word"
        if tok in {",", ";", ":", "?", "!"}:
            return "punct"
        return "math"

    def remove_math_suffix(prefix: str) -> str:
        """
        Delete consecutive "math tokens" at the end of the prefix,
        keeping the last word or punctuation and the space after it.
        """
        spans = []
        n = len(prefix)
        i = 0
        while i < n:
            if prefix[i].isspace():
                i += 1
                continue
            j = i
            while j < n and not prefix[j].isspace():
                j += 1
            tok = prefix[i:j]
            ttype = token_type(tok)
            spans.append((i, j, ttype))
            i = j
        if not spans:
            return prefix

        k = len(spans) - 1
        while k >= 0 and spans[k][2] == "math":
            k -= 1
        if k < 0:
            cut_pos = 0
        else:
            cut_pos = spans[k][1]
            while cut_pos < n and prefix[cut_pos].isspace():
                cut_pos += 1
        return prefix[:cut_pos]
    key = '{\\displaystyle'
    if key not in line:
        return line
    
    spans = find_displaystyle_spans(line)
    if not spans:
        return line
    out = []
    idx = 0
    for start, end, inside in spans:
        prefix = line[idx:start]
        prefix2 = remove_math_suffix(prefix)
        out.append(prefix2)
        if prefix2 and not prefix2[-1].isspace():
            out.append(" ")
        out.append(f"${inside.strip()}$")
        idx = end
    out.append(line[idx:])
    return "".join(out)


def clean_wiki_math_with_latex(lines, max_prefix_span: int = 80, sim_th: float = 0.7):
    """
    Process the list of texts scraped from Wikipedia:
      - Extract formulas inside {\\displaystyle ...}
      - Delete repetitive duplicate ASCII formulas (such as Hom _ (h x, F), h x, C ↓ x, etc.) preceding it
      - Insert the LaTeX formula in the form of $formula$ into the text

    Args:
      lines          : List[str]  Original list of strings (one string per paragraph/line)
      max_prefix_span: Look for "repetitive formulas" only within the last N characters before the formula
      sim_th         : Similarity threshold between ASCII and LaTeX formulas (larger is more conservative)

    Returns:
      cleaned_lines: List[str]  Cleaned text
      all_formulas : List[str]  All '$formula$' (convenient to use separately)
    """
    def normalize_math(s: str) -> str:
        """
        Rough normalization of the math string:
        - Remove LaTeX commands (like \\operatorname)
        - Keep only letters and numbers
        Used to compare the similarity between "ASCII formula" and LaTeX formula.
        """
        s2 = re.sub(r"\\[a-zA-Z]+", "", s)
        s2 = re.sub(r"[^A-Za-z0-9]+", "", s2)
        return s2.lower()

    def extract_display_blocks(text: str):
        """
        Find all {\\displaystyle ...} blocks in text:
        Returns [(start, end, formula), ...]
          - start: start position of '{\\displaystyle'
          - end  : next position after the matching outer '}'
          - formula: content removing {\\displaystyle and the outermost curly braces
        """
        blocks = []
        i = 0
        n = len(text)
        while True:
            start = text.find(DISPLAY_START, i)
            if start == -1:
                break
            depth = 1
            k = start + 1
            while k < n and depth > 0:
                c = text[k]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                k += 1
            end = k  
            formula = text[start + len(DISPLAY_START): end - 1].strip()
            blocks.append((start, end, formula))
            i = end
        return blocks

    def clean_line(line: str):
        blocks = extract_display_blocks(line)
        if not blocks:
            return line, []
        parts = []
        formulas_latex = []
        last_idx = 0
        for start, end, formula in blocks:
            prefix = line[last_idx:start]
            window_start = max(0, len(prefix) - max_prefix_span)
            norm_formula = normalize_math(formula)
            best_sim = 0.0
            best_cut = len(prefix)
            if norm_formula:
                for i in range(window_start, len(prefix) + 1):
                    cand = prefix[i:]
                    norm_cand = normalize_math(cand)
                    if not norm_cand:
                        continue
                    sim = SequenceMatcher(None, norm_cand, norm_formula).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_cut = i
            cut_idx = len(prefix)
            if best_sim >= sim_th:
                cut_idx = best_cut
            left = prefix[:cut_idx].rstrip()
            if left:
                parts.append(left)
                if not left[-1].isspace() and left[-1] not in "([{":
                    parts.append(" ")
            latex_formula = f"${formula}$"
            parts.append(latex_formula)
            formulas_latex.append(latex_formula)
            last_idx = end
        suffix = line[last_idx:]
        parts.append(suffix)
        return "".join(parts), formulas_latex
    
    cleaned_lines = []
    all_formulas = []
    for line in lines:
        new_line, fs = clean_line(line)
        cleaned_lines.append(new_line)
        all_formulas.extend(fs)
    return cleaned_lines, all_formulas


def pre_clean_wikipedia(paragraphs):
    pre_marker = "From Wikipedia, the free encyclopedia"
    if pre_marker in paragraphs:
        idx = paragraphs.index(pre_marker)
        paragraphs = paragraphs[idx + 1:]
    last_markers = ["References", "External links", "See also", "Further reading", "Notes"]
    last_idx = len(paragraphs)
    for i in range(len(paragraphs) - 1, -1, -1):
        p = paragraphs[i].strip()
        if p in last_markers and last_idx > i:
            last_idx = i
    if last_idx < len(paragraphs):
        paragraphs = paragraphs[:last_idx]
    return paragraphs


def fetch_and_clean_html(url: str, return_str: bool = True):
    """
    Only responsible for:
    1) Fetching the raw HTML of the webpage
    2) Removing HTML tags to get plain text
    Does not do any content filtering, similarity checking, paragraph extraction, wiki special case handling, etc.
    """
    try:
        html = fetch_page_text(url)  
    except Exception:
        return "" if return_str else []
    if not html:
        return "" if return_str else []
    try:
        text = extract_text_from_html(html)  
    except Exception:
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
    text = (text or "").strip()
    if return_str:
        return text
    return [line for line in text.splitlines() if line.strip()]


def fetch_and_clean(url, resiliparse = None, stop_words = None, return_str: bool = True):
    if resiliparse is None:
        resiliparse = ResiliparseExtractor(
            required_stopword_density=0.0, main_content=False  # Only keep paragraphs with >= 32% stop words
        )
    if stop_words is None:
        stop_words = frozenset(["the", "and", "is", "in", "for", "where", "when", "to", "at"])  # 
    text = fetch_page_text(url)
    try:
        paragraphs = resiliparse.extract_text(text, stop_words, language="ENGLISH")
    except Exception as e:
        paragraphs = extract_text_from_html(text) if return_str else [extract_text_from_html(text)]

    if "wikipedia.org" in url:
        paragraphs = pre_clean_wikipedia(paragraphs) 
    
    cleaned_paragraphs = clean_paragraphs(paragraphs, stopwords=stop_words, keep_formula_window=3, min_sentence_len=20, url=url)
    cleaned_paragraphs, formulas = clean_wiki_math_with_latex(cleaned_paragraphs, max_prefix_span=300)
    first_idx = float('inf')
    for idx, clean in enumerate(cleaned_paragraphs):
        if "http" in clean or "www." in clean:
            first_idx = min(first_idx, idx)

    cleaned_paragraphs = cleaned_paragraphs[:first_idx] if first_idx != float('inf') else cleaned_paragraphs
    if return_str:
        return "\n".join(cleaned_paragraphs)
    return cleaned_paragraphs
    

    