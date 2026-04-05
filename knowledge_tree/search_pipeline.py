import os
import json
import requests
from typing import Union, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .clean_utils import fetch_and_clean
from .worker_web_high_eff import validate_language
import re
import copy
import time

DEEPCRAWL_READ_API = os.environ.get("DEEPCRAWL_READ_API", "https://api.deepcrawl.dev/read")

SEARCHCANS_API = os.environ.get("SEARCHCANS_API", "https://searchcans.youxikuang.cn/api/search")
SEARCHCANS_TOKEN = os.environ.get("SEARCHCANS_TOKEN")
DEEPCRAWL_API_KEY = os.environ.get("DEEPCRAWL_API_KEY")

def search_and_fetch_grouped(
    keywords: Union[str, List[str]],
    *,
    engine: str = "bing",
    page: int = 1,
    timeout_ms: int = 10000,
    keyword_workers: int = 4,
    fetch_workers: int = 10,
    per_keyword_max_items: int = 50,
    searchcans_token: Optional[str] = None,
    deepcrawl_api_key: Optional[str] = None,
    temp_search_path: Optional[str] = None,  # ✅ TODO: translate comment
) -> List[Dict[str, Any]]:
    """
    每个关键词单独搜索 -> 抓取每条结果的网页正文（markdown） -> 校验 -> 保存(JSONL)。
    返回过滤后的结果列表。
    """
    searchcans_token = searchcans_token or SEARCHCANS_TOKEN
    deepcrawl_api_key = deepcrawl_api_key or DEEPCRAWL_API_KEY
    if not searchcans_token:
        raise RuntimeError("Missing SEARCHCANS_TOKEN")
    if not deepcrawl_api_key:
        raise RuntimeError("Missing DEEPCRAWL_API_KEY")

    kw_list = [keywords] if isinstance(keywords, str) else keywords
    if not kw_list:
        return []

    s_headers = {"Authorization": f"Bearer {searchcans_token}", "Content-Type": "application/json"}

    def search_one(kw: str) -> List[Dict[str, Any]]:
        payload = {"s": kw, "t": engine, "p": page, "d": timeout_ms}
        max_attempts = 1
        for attempt in range(1, max_attempts+1):  # TODO: translate comment
            try:
                r = requests.post(SEARCHCANS_API, headers=s_headers, json=payload, timeout=timeout_ms)
                r.raise_for_status()
                j = r.json()
                if j.get("code") != 0:
# TODO: translate comment
                    raise RuntimeError(f"SearchCans error: code={j.get('code')} msg={j.get('msg')}")

                items = []
                for it in (j.get("data") or [])[:per_keyword_max_items]:
                    title = (it.get("title") or "").strip()
                    url = (it.get("url") or "").strip()
                    if title and url and url.startswith(("http://", "https://")):
                        items.append({"keyword": kw, "title": title, "url": url})
                return items

            except Exception:
# TODO: translate comment
                if attempt < max_attempts:
                    time.sleep(1.0)
                    continue
                return []
        
        
        # r = requests.post(SEARCHCANS_API, headers=s_headers, json=payload, timeout=timeout_ms)
        # r.raise_for_status()
        # j = r.json()

        # if j.get("code") != 0:
        #     raise RuntimeError(f"SearchCans error: code={j.get('code')} msg={j.get('msg')}")

        # items = []
        # for it in (j.get("data") or [])[:per_keyword_max_items]:
        #     title = it.get("title") or ""
        #     url = it.get("url") or ""
        #     if title and url and url.startswith(("http://", "https://")):
        #         items.append({"keyword": kw, "title": title, "url": url})
        # return items

    def deepcrawl_read(url: str) -> str:
        r = requests.get(
            DEEPCRAWL_READ_API,
            params={"url": url},
            headers={"Authorization": f"Bearer {deepcrawl_api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        return r.text  # markdown TODO: translate comment

    def fetch_one(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:       
        kw = item["keyword"]
        title = item["title"]
        url = item["url"]

# 1) DeepCrawl TODO: translate comment
        try:
            content_md = deepcrawl_read(url)
            if "No content could be extracted from this URL" in content_md or "Protected or restricted content" in content_md:
                print(f"========= deepcrawl由于规则限制, 抽取失败, url:{url} =========")
                try:
                    content_md = fetch_and_clean(url)
                except Exception:
                    return None 
        except Exception:
            print(f"========= deepcrawl发生错误, 抽取失败, url:{url} =========")
            # 2) fallback
            try:
                content_md = fetch_and_clean(url)
            except Exception:
                return None

        if (
            not content_md
            or "[请求失败]" in content_md
            or len(content_md.strip()) < 200
            or not validate_language(content_md, type="content", link=url)
        ):
# print(f"=============== TODO: translate comment
            # print(content_md)
            # print("=============================================================")
            return None
        content = strip_all_markdown_links(content_md)
        return {"keyword": kw, "title": title, "url": url, "content": content}

# 1) TODO: translate comment
    items_by_kw: Dict[str, List[Dict[str, Any]]] = {kw: [] for kw in kw_list}
    with ThreadPoolExecutor(max_workers=min(keyword_workers, len(kw_list))) as ex:
        futs = {ex.submit(search_one, kw): kw for kw in kw_list}
        with tqdm(total=len(futs), desc="Search keywords", unit="kw") as pbar:
            for fut in as_completed(futs):
                kw = futs[fut]
                try:
                    items_by_kw[kw] = fut.result()
                except Exception as e:
                    print(f"[search] keyword={kw} failed: {e}", flush=True)
                    items_by_kw[kw] = []
                finally:
                    pbar.update(1)

# ✅ 1.5) TODO: translate comment
    if temp_search_path:
        with open(temp_search_path, "w", encoding="utf-8") as f:
            json.dump(items_by_kw, f, ensure_ascii=False, indent=2)
        print(f"[temp] saved search results -> {temp_search_path}", flush=True)

    web_search_results = copy.deepcopy(items_by_kw)
    
# 2) TODO: translate comment
    for kw, items in items_by_kw.items():
        seen = set()
        uniq = []
        for it in items:
            u = it["url"]
            if u not in seen:
                seen.add(u)
                uniq.append(it)
        items_by_kw[kw] = uniq

# 3) TODO: translate comment
    grouped: Dict[str, List[Dict[str, str]]] = {kw: [] for kw in kw_list}
    all_items = [it for items in items_by_kw.values() for it in items]

    with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
        future_to_item = {ex.submit(fetch_one, item): item for item in all_items}
        with tqdm(total=len(future_to_item), desc="Fetch pages", unit="url") as pbar:
            for fut in as_completed(future_to_item):
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[fetch_one] failed: {e}", flush=True)
                    result = None
                finally:
                    pbar.update(1)

                if result is None:
                    continue
                kw = result["keyword"]
                grouped[kw].append({"url": result["url"], "title": result["title"], "content": result["content"]})

# TODO: translate comment
    for kw in kw_list:
        print(f"{kw}: {len(grouped[kw])} / {len(items_by_kw[kw])}", flush=True)

    return grouped, web_search_results

def strip_all_markdown_links(md: str) -> str:
    """
    删除 Markdown 中所有超链接（含图片链接、嵌套链接）。
    - [text](url) -> text
    - ![alt](url) -> alt
    - [![alt](...)](url) -> alt
    不保留任何 URL。
    """
    if not md:
        return md
    md = md.replace("\u00a0", " ")  # \xa0 -> space
# TODO: translate comment
    md = re.sub(r"\[\s*(!\[[^\]]*\]\([^)]+\))\s*\]\([^)]+\)", r"\1", md)
# TODO: translate comment
    prev = None
    while prev != md:
        prev = md
# TODO: translate comment
        md = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", md)
# TODO: translate comment
# text TODO: translate comment
        md = re.sub(r"\[([\s\S]*?)\]\([^)]+\)", r"\1", md)
# TODO: translate comment
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


if __name__ == "__main__":
    res = search_and_fetch_grouped(
        ["Inhibition of Na+/K+-ATPase by cardiac glycosides", "Definition: Radiographic Field of View"],
        per_keyword_max_items=10,
        keyword_workers=3,
        fetch_workers=12,
    )
    print(res)
    
    