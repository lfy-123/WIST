#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple SearXNG search client.

依赖:
    pip install requests
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Any, Dict

import requests
import json
from requests.exceptions import ReadTimeout, RequestException
import os

# Define the OpenAPI server URL
API_URL = "https://gpts.webpilot.ai/api/read"
WEB_UID = "b7511b4261a04453b085c369ba0a1d8f"


# -----------------------------------------------------------------------------
# TODO: translate comment
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
# TODO: translate comment
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# TODO: translate comment
# -----------------------------------------------------------------------------
@dataclass
class SearchResult:
    """单条搜索结果"""

    link: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # TODO: translate comment


# -----------------------------------------------------------------------------
# TODO: translate comment
# -----------------------------------------------------------------------------
def search_searxng(
    query_url: str,
    query: str,
    count: int = 10,
    filter_list: Optional[list[str]] = None,
    **kwargs,
) -> List[SearchResult]:
    """
    调用 SearXNG 实例进行搜索，并返回 SearchResult 列表。

    参数:
        query_url (str):
            SearXNG 的搜索接口 URL，一般形如:
                http://127.0.0.1:8888/search
            或者某些公共实例:
                https://searxng.example.com/search

        query (str):
            搜索关键词，如 "Hodge index theorem"

        count (int):
            返回结果数量上限（本函数在客户端侧截断）

        filter_list (list[str] | None):
            兼容原函数的参数，这里不再使用。
            你可以自己扩展逻辑，比如按域名过滤等。

    可选 Keyword Args (**kwargs):
        language (str):
            语言，例如 "en-US"、"zh-CN"，默认 "en-US"

        safesearch (str | int):
            安全搜索等级：0=关闭, 1=中等, 2=严格，默认 "1"

        time_range (str):
            时间范围过滤，比如 "day" / "week" / "month" / "year"，
            或者具体格式视你 SearXNG 配置而定。默认 "" 不限制。

        categories (list[str]):
            类别列表，例如 ["general"]、["science"] 等。
            在请求中会拼成一个字符串传给 SearXNG。

        timeout (float):
            requests.get 的超时时间（秒），默认 10.0

    返回:
        List[SearchResult]:
            按 score 字段从高到低排序后的结果（若无 score，则按原顺序）。

    可能抛出的异常:
        requests.exceptions.RequestException:
            网络错误、超时、HTTP 错误等。
        ValueError:
            返回内容不是 JSON，或者缺少必要字段。
    """

# TODO: translate comment
    language = kwargs.get("language", "en-US")
    safesearch = kwargs.get("safesearch", "1")
    time_range = kwargs.get("time_range", "")
    categories = "".join(kwargs.get("categories", []))
    timeout = kwargs.get("timeout", 10.0)

# TODO: translate comment
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "safesearch": safesearch,
        "language": language,
        "time_range": time_range,
        "categories": categories,
        "theme": "simple",
        "image_proxy": 0,
    }

# TODO: translate comment
    if "<query>" in query_url:
        query_url = query_url.split("?", 1)[0]

    logger.debug("searching %s with params=%s", query_url, params)

    response = requests.get(
        query_url,
        headers={
# TODO: translate comment
            # "User-Agent": "SimpleSearxngClient/1.0 (+https://example.com)",
            # "Accept": "text/html",
            "User-Agent": "MySearxngClient/0.1 (+https://example.com)",
            "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
            # "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        },
        params=params,
        timeout=timeout,
    )

# HTTP TODO: translate comment
    response.raise_for_status()

    try:
        # print(f"[search_searxng] response text: {response.text[:500]}...")
        json_response = response.json()
    except ValueError as exc:
        logger.error("Response is not valid JSON: %s", exc)
        raise

    results = json_response.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Unexpected response format: 'results' is not a list")

# TODO: translate comment
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

# TODO: translate comment
# TODO: translate comment
    # if filter_list:
    #     sorted_results = [
    #         r for r in sorted_results
    #         if any(f in r.get("url", "") for f in filter_list)
    #     ]

    search_results: List[SearchResult] = []
    for result in sorted_results[:count]:
        url = result.get("url")
        if not url:
# SearXNG TODO: translate comment
            continue

        search_results.append(
            SearchResult(
                link=url,
                title=result.get("title"),
                snippet=result.get("content"),
                raw=result,
            )
        )

    return search_results



def show_path(path):
    idx = 0
    if path[0].name == "Root":
        idx = 1
    names = []
    for node in path[idx:]:
        names.append(node.name)
    return(' -> '.join(names))

def split_path(path_str):
    names = path_str.split(' -> ')
    return names


def search_web_content(link, keyword='', language="en"):
    # Prepare the request payload
    payload = {
        "link": link,  
        "ur": keyword,  # The user's request (search query)
        "lp": True,  # Whether the link is directly provided by the user (True for now)
        "rt": False,  # Retry flag (set to False for first request)
        "l": language  # Language code (e.g., "zh-CN" for Simplified Chinese)
    }

    # Set headers for JSON request
    headers = {
        'Content-Type': 'application/json',
        'WebPilot-Friend-UID': WEB_UID
    }

    # Make the POST request to the WebPilot API
    response = requests.post(API_URL, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        # Extract and display relevant information
        # print("Page Title:", data.get('title'))
        # print("Page Content:", data.get('content'))
        # print(f"data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return data
    else:
        # print(f"Error: {response.status_code} - {response.text}")
        return f"Error: {response.status_code} - {response.text}"



import requests
from bs4 import BeautifulSoup

def fetch_full_page_text(url: str) -> str:
    resp = requests.get(
        url,
        headers={
            "User-Agent": "MyCrawler/0.1 (+https://example.com)",
        },
        timeout=15,
    )
    resp.raise_for_status()
    html = resp.text
    try:
# TODO: translate comment
        soup = BeautifulSoup(html, "lxml")
    except ValueError:
# TODO: translate comment
        soup = BeautifulSoup(html, "html.parser")
        
    # soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)



def clean_html(file_path: str) -> str:
# TODO: translate comment
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()

# TODO: translate comment
    soup = BeautifulSoup(html, 'lxml')

# 1. TODO: translate comment
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

# 2. TODO: translate comment
    for tag in soup(['header', 'footer', 'nav', 'aside']):
        tag.decompose()

# 3. TODO: translate comment
    text = soup.get_text(separator='\n')

# 4. TODO: translate comment
    lines = [line.strip() for line in text.splitlines()]
# TODO: translate comment
    lines = [line for line in lines if line]
    clean_text = "\n".join(lines)

    return clean_text




def clean_html_text(html: str) -> str:
    """对一段 HTML 文本做清洗，返回纯文本。"""

# TODO: translate comment
    soup = BeautifulSoup(html, "lxml")

# 1. TODO: translate comment
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

# 2. TODO: translate comment
    for tag in soup(["header", "footer", "nav", "aside"]):
        tag.decompose()

# 3. TODO: translate comment
    text = soup.get_text(separator="\n")

# 4. TODO: translate comment
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    clean_text = "\n".join(lines)

    return clean_text


def fetch_and_clean(url: str, timeout: float = 15.0) -> str:
    """类似 curl -L URL | clean_html，直接返回清洗后的纯文本。"""

    headers = {
# TODO: translate comment
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(
            url,
            headers=headers,
            allow_redirects=True,  # TODO: translate comment
            timeout=timeout,
        )
        resp.raise_for_status()
    except ReadTimeout:
        return f"[请求失败] 读取网页超时，跳过：{url}"   # TODO: translate comment
    except RequestException as e:
# TODO: translate comment
        return f"[请求失败] {url} - {e}"

# TODO: translate comment
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"

    html = resp.text
    return clean_html_text(html)


# ====================== TODO: translate comment
def load_web_corpus(path: str) -> List[Dict]:
    """
    从指定路径读取 wiki 语料文件。
    约定结构：{"title": {"content":..., ...}, "title": {"content":..., ...}, ...}
    如果文件不存在或内容非法，返回空列表。
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[web_worker] Failed to load wiki corpus from {path}, error: {e}")
        return {}
    return data


def save_web_corpus(path: str, entries: List[Dict]) -> None:
    """
    将 wiki 语料列表写回指定路径。
    为安全起见，采用临时文件覆盖写。
    """
    if os.path.exists(path):
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
            print(f"[web_worker] Wiki corpus saved to {path}, total entries: {len(entries)}")
        except Exception as e:
            print(f"[web_worker] Failed to save wiki corpus to {path}, error: {e}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)





import re
def _validate_proposal(proposal: str) -> bool:
    """Validate a proposed domain name with Title Case normalization"""
    if not proposal:  # Check for empty or overly long proposals
        return False
# TODO: translate comment
    proposal = proposal.strip()
# TODO: translate comment
    if len(proposal) == 0 or len(proposal) > 100 or proposal == "and":
        return False
    if re.search(r'[\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]', proposal):
        print(f"[验证失败] proposal 包含非英文 CJK/韩文/阿拉伯字符: {proposal}")
        return False
# ========= TODO: translate comment
# TODO: translate comment
    ALLOWED_PUNCT_ASCII = set("-_',:()^*/+=.·\\[]{}|<>~")
# TODO: translate comment
    ALLOWED_EXTRA_RANGES = [
        (0x2070, 0x209F),  # superscript/subscript
        (0x2200, 0x22FF),  # math symbols
        (0x0391, 0x03A9),  # Greek uppercase
        (0x03B1, 0x03C9),  # Greek lowercase
    ]
# TODO: translate comment
    ALLOWED_EXTRA_CHARS = {
        "\u2032",  # ′ prime
        "–",       # U+2013 en dash
        "—",       # U+2014 em dash
    }
    for ch in proposal:
# TODO: translate comment
        if ch.isspace():
            continue
        code = ord(ch)
# 1) ASCII TODO: translate comment
        if ch.isascii():
            if ch.isalnum() or ch in ALLOWED_PUNCT_ASCII:
                continue
            print(f"[验证失败] proposal 包含非常规 ASCII 字符: {repr(ch)} in {proposal}")
            return False
# 2) TODO: translate comment
# TODO: translate comment
        if ch.isalpha():
            continue
# 3) TODO: translate comment
        if ch in ALLOWED_EXTRA_CHARS:
            continue
# 4) TODO: translate comment
        if any(start <= code <= end for (start, end) in ALLOWED_EXTRA_RANGES):
            continue
# 5) TODO: translate comment
        print(f"[验证失败] proposal 包含非常规非 ASCII 字符: {repr(ch)} in {proposal}")
        return False
    lower = proposal.lower()
# 2. TODO: translate comment
    forbidden_substrings = [
        "proposed",   # TODO: translate comment
        "proposal",   # TODO: translate comment
        "completed",  # "generation completed" TODO: translate comment
        "assistant",  # TODO: translate comment
        "user",       # TODO: translate comment
        "system",     # TODO: translate comment
        "next level", # TODO: translate comment
        "corrected",
        "confirmed",
        "12x8",       # TODO: translate comment
    ]
    if any(s in lower for s in forbidden_substrings):
        return False
    if re.search(r"[.;!\?\[\]\{\}]", proposal):
        return False
# 4. TODO: translate comment
    tokens = re.findall(r"[a-z0-9']+", lower)
    forbidden_words = {"unk", "unknown", "none", "maybe"}
    if any(tok in forbidden_words for tok in tokens):
        return False
    return True


# -----------------------------------------------------------------------------
# TODO: translate comment
# -----------------------------------------------------------------------------
if __name__ == "__main__":
# TODO: translate comment
    searxng_url = "http://127.0.0.1:8888/search"
    query = "The Definition of p-adic Exponentiation"

    try:
        res = search_searxng(
            query_url=searxng_url,
            query=query,
            count=5,
            language="en",
            safesearch="0",
            categories=["general"],
        )
        for i, r in enumerate(res, start=1):
            print(f"[{i}] {r.title}")
            print(f"    {r.link}")
            if r.snippet:
                print(f"    snippet: {r.snippet[:120]}...")
            if r.raw:
                print(f"    raw: {r.raw}...")
            print()
    except requests.RequestException as e:
        logger.error("Request to SearXNG failed: %s", e)
