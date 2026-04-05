import os
import re
import time
import argparse
import json
import glob
import certifi
import requests
import wikipedia
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

from .clean_utils import fetch_and_clean
from .utils import load_web_corpus, save_web_corpus, search_searxng

MWMBL_BASE = "https://mwmbl.org"

def _normalize_mwmbl_title(title_obj) -> str:
    if title_obj is None:
        return ""
    if isinstance(title_obj, str):
        return title_obj.strip()
    if isinstance(title_obj, list):
        parts = []
        for seg in title_obj:
            if isinstance(seg, dict):
                parts.append(str(seg.get("value", "")))
            else:
                parts.append(str(seg))
        return "".join(parts).strip()
    return str(title_obj).strip()



def search_mwmbl(query: str, count: int = 50, timeout: int = 20) -> List[Dict]:
    """
    Use the public search API of mwmbl.org to search the web.
    The returned structure is unified as:
      [{"title": str, "link": str, "engine": "mwmbl"}, ...]
    Any exceptions are caught internally and return an empty list to avoid affecting the main process.
    """
    try:
        url = f"{MWMBL_BASE}/api/v1/search/"
        headers = {"User-Agent": "mwmbl-client/1.0 (+https://example.invalid)"}
        r = requests.get(url, params={"s": query}, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[search_mwmbl] failed: {e}", flush=True)
        return []

    results = data if isinstance(data, list) else data.get("results", [])
    out: List[Dict] = []
    for item in results[:max(1, count)]:
        try:
            title = _normalize_mwmbl_title(item.get("title"))
            link = (item.get("url") or item.get("href") or "").strip()
            if not title or not link:
                continue
            out.append({"title": title, "link": link, "engine": "mwmbl"})
        except Exception:
            continue
    return out


EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/inspire/hdd/global_user/lifangyuan-253108110077/Models/all-MiniLM-L6-v2")
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_PATH)

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def normalize_wiki_title(s: str) -> str:
    """
    Convert a string to "Title Case":
    - Capitalize the first letter of each word
    - Except for the first word, keep small words like and / of / in / on / at / for / to / a / an / the / by / from / with / as lowercase
    """
    if not s:
        return s
    s = s.strip()
    if not s:
        return s
    small_words = {
        "and", "or", "of", "in", "on", "at", "for", "to",
        "a", "an", "the", "by", "from", "with", "as"
    }
    titled = s.title()
    parts = titled.split()
    new_parts = []
    for i, w in enumerate(parts):
        if i > 0 and w.lower() in small_words:
            new_parts.append(w.lower())
        else:
            new_parts.append(w)
    return " ".join(new_parts)


def wiki_search_full(domain: str) -> Dict[str, str]:
    """
    Search for 'domain' on wikipedia, return a dictionary of {Entry Title: Full Content String}.
    The content is simply cleaned, but not chunked.
    """
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            results = wikipedia.search(domain)
            break
        except Exception as e:
            print(
                f"[wiki_search_full] Error searching '{domain}' "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                time.sleep(1)
        if attempt == max_retries:
            return {}, {}
        

    print(f"[wiki_search_full] Searching for: {domain}, results: {results}")
    if not results:
        return {}, {}

    contents: Dict[str, str] = {}
    urls: Dict[str, str] = {}
    for result in results:
        try:
            page = wikipedia.page(result, auto_suggest=False)
            if not page.content or not page.content.strip():
                print(f"[wiki_search_full][Warning] {result} has no content.")
                continue

            text = page.content
            # text = filter_content(text)
            # text = add_newlines_around_headings(text)

            norm_title = normalize_wiki_title(result)
            if text.strip():
                contents[norm_title] = text
                urls[norm_title] = page.url
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[wiki_search_full][Disambiguation] {result} -> options: {e.options[:3]}...")
        except wikipedia.exceptions.PageError:
            print(f"[wiki_search_full][PageError] {result} not found.")
        except Exception as e:
            print(f"[wiki_search_full][Error] {result} failed with error: {e}")
    return contents, urls  # {TODO: translate comment


def process_level_with_wiki(to_query, level: int, existing_titles, max_workers: int = 4):
    """
    Perform a wiki query on a specific level of the given KnowledgeTree:
    - Only process nodes with wiki_status == "unqueried" or without the wiki_status field;
    - Store only lightweight information in nodes:
        node.wiki_titles = [title1, title2, ...]
        node.wiki_status = "queried" / "no_result"
        node.wiki_valid = None
    - The actual wiki text content is unified and stored in an independent file web_corpus_path
        [
          {"title": "...", "content": "..."},
          {"title": "...", "content": "..."},
          ...
        ]
      Every time it is called:
        1. Read the old file first;
        2. Append the newly found entries;
        3. Avoid duplicate titles;
        4. Overwrite the original file.
    """
    # wiki_entries = load_web_corpus(web_corpus_path)
    # existing_titles = set(wiki_entries.keys())
    final_results = {}
    new_wiki_entries = {}
    
    if not to_query:
        return final_results, new_wiki_entries
    
    print(
        f"[web_worker] process_level_with_wiki: level={level}, "
        f"nodes_to_query={len(to_query)}",
        flush=True,
    )
    
    def worker(name: str):
        print(f"[web_worker] Querying web for node '{name}' (level={level})")
        contents, urls = wiki_search_full(name)  # {title: full_text}
        return name, contents, urls
    
    max_workers = min(max_workers, len(to_query))
    if max_workers <= 0:
        return final_results, new_wiki_entries
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(worker, name): name for name in to_query
        }
        for fut in as_completed(future_to_name):
            name, contents, urls = fut.result()
            if contents:
                final_results[name] = {
                    "wiki_titles": list(contents.keys()),
                    "wiki_status": "queried",
                    "wiki_valid": None,
                    "beta_param": (1, 1),
                }

                for title, full_text in contents.items():
                    url = urls[title]
                    title = (title or "").strip()
                    if not title:
                        continue
                    if title in existing_titles:
                        continue
                    new_wiki_entries[title] = {
                        "content": (full_text or "").strip(),
                        "url": url,
                    }
                    # existing_titles.add(title)
                    print(f"[web_worker] Added wiki entry: {title}")
            else:
                final_results[name] = {
                    "wiki_titles": [],
                    "wiki_status": "no_result",
                    "wiki_valid": None,
                    "beta_param": (1, 1e7),
                }
    # save_web_corpus(web_corpus_path, wiki_entries)
    return final_results, new_wiki_entries


def process_level_with_search(to_query, level, existing_titles, count=100):
    """
    Perform a web query on the last level of the given KnowledgeTree:
    - Only process nodes with wiki_status == "unqueried" or without the wiki_status field;
    - Store only lightweight information in nodes:
        node.wiki_titles = [title1, title2, ...]
        node.wiki_status = "queried" / "no_result"
        node.wiki_valid = None
    - The actual web text content is unified and stored in an independent file web_corpus_path
        {
          "title1": {"content": "...", "url": "...", "similarity": ...},
          "title2": {...},
          ...
        }
      Every time it is called:
        1. Read the old file first;
        2. Append the newly found entries (deduplicate by title);
        3. Overwrite the original file.
    """
    # web_entries = load_web_corpus(web_corpus_path)
    # existing_titles = set(web_entries.keys())
    final_results = {}
    new_web_entries = {}
    if not to_query:
        return final_results, new_web_entries

    print(
        f"[web_worker] process_level_with_search: level={level}, "
        f"nodes_to_query={len(to_query)}",
        flush=True,
    )

    def worker(name: str):
        print(f"[web_worker] Querying web for node '{name}' (level={level})")
        all_results = get_all_search_content(
            name,
            count=count,
            sim_threshold=0.5,
            discard_ratio=0.5,
        )
        return name, all_results

    max_workers = min(8, len(to_query))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(worker, name): name for name in to_query}
        for fut in as_completed(future_to_name):
            name, all_results = fut.result()

            if all_results:
                all_titles = [item["title"].strip() for item in all_results]
                final_results[name] = {
                    "wiki_titles": all_titles,
                    "wiki_status": "queried",
                    "wiki_valid": None,
                    "beta_param": (1, 1),
                }
                for result in all_results:
                    title = (result.get("title") or "").strip()
                    if not title:
                        continue
                    if title in existing_titles:
                        continue
                    full_text = (result.get("content") or "").strip()
                    link = result.get("link", "")
                    similarity = result.get("similarity", 0.0)
                    new_web_entries[title] = {
                        "content": full_text,
                        "url": link,
                        "similarity": similarity,
                    }
                    # existing_titles.add(title)
                    print(f"[web_worker] Added web entry: {title}")
            else:
                final_results[name] = {
                    "wiki_titles": [],
                    "wiki_status": "no_result",
                    "wiki_valid": None,
                    "beta_param": (1, 1e7),
                }

    # save_web_corpus(web_corpus_path, web_entries)
    return final_results, new_web_entries


def process_level_with_web(to_query, level: int, max_levels: int, existing_titles, count: int = 100):
    if level == max_levels:
        return process_level_with_search(to_query, level, existing_titles, count=count)
    else:
        return process_level_with_wiki(to_query, level, existing_titles)    
    

def validate_url(url: str, filter_url: list) -> bool:
    if any(filt.lower() in url.lower() for filt in filter_url):
        print(f"[Validation Failed] URL Filtering: {url}")
        return False
    return True



def validate_language(content: str, type: str = "title", link = None) -> bool:
    if type == "title":
        return _validate_title_language(content, link=link)
    else:
        snippet = (content or "")[:4000]
        return _validate_content_language(snippet, link=link)



# Common non-Latin scripts used to identify clearly non-English text.
NON_LATIN_SCRIPTS_PATTERN = re.compile(
    "[" 
    "\u4e00-\u9fff"
    "\u3040-\u30ff"
    "\u31f0-\u31ff"
    "\u1100-\u11ff"
    "\u3130-\u318f"
    "\uac00-\ud7af"
    "\u0600-\u06ff"
    "\u0750-\u077f"
    "\u08a0-\u08ff"
    "\u0400-\u04ff"
    "]"
)

HAS_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")  # CJK unified ideographs
def _validate_title_language(title: str, link) -> bool:
    if not title:
        print(f"[Validation Failed] Title is empty, url:{link}")
        return False
    title = title.strip()
    if not title:
        print(f"[Validation Failed] Title is all whitespace, url:{link}")
        return False
    if HAS_CJK_PATTERN.search(title):
        print(f"[Validation Failed] Title contains Chinese: {title}, url:{link}")
        return False
    ascii_letters = sum("a" <= ch.lower() <= "z" for ch in title)
    non_latin_chars = len(NON_LATIN_SCRIPTS_PATTERN.findall(title))
    if non_latin_chars > 0 and ascii_letters <= non_latin_chars:
        print(f"[Validation Failed] Title is suspected to be non-English native language: {title}, url:{link}")
        return False
    return True

EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "for", "on", "with", "as", "by",
    "from", "that", "this", "are", "be", "an", "at", "or", "it", "if", "which",
    "we", "can", "has", "have", "not", "such", "then", "also", "but", "one",
    "all", "any", "each", "other", "more", "no", "may", "so", "than", "there",
    "their", "them", "these", "those", "our", "its", "into", "about", "over",
}



    

def _validate_content_language(text: str, link) -> bool:
    """
    Determine if the "Main Language" of a piece of text is English:
      1. If the proportion of non-Latin scriptures is too high -> non-English
      2. Otherwise, use the English stop word ratio to determine if it is clearly English
    """
    if not text:
        print(f"[Validation Failed] Content is empty, url:{link}")
        return False
    if HAS_CJK_PATTERN.search(text):
        print(f"[Validation Failed] Content contains Chinese: {text[:100]}, url:{link}")
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters == 0:
        print(f"[Validation Failed] No letters in the content, unable to judge language, viewed as non-English, url:{link}")
        return False
    non_latin_chars = len(NON_LATIN_SCRIPTS_PATTERN.findall(text))
    non_latin_ratio = non_latin_chars / letters
    # If the non-Latin ratio is very high, treat the text as non-English.
    if non_latin_ratio > 0.3:
        print(f"[Validation Failed] Proportion of non-Latin scriptures is too high ({non_latin_ratio:.2f}), suspected non-English, url:{link}")
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    # Very short text is hard to score reliably, so we avoid over-filtering here.
    # This can be changed to False if stricter filtering is desired.
    if len(words) < 300:
        return False
    sw_count = sum(w in EN_STOPWORDS for w in words)
    sw_ratio = sw_count / len(words)
    # English text usually contains a moderate proportion of stopwords.
    # Other Latin-script languages usually have a much lower English stopword ratio.
    if sw_ratio < 0.15:
        print(f"[Validation Failed] Content English stop word proportion too low ({sw_ratio:.3f}), suspected non-English, url:{link}")
        return False
    # If execution reaches this point, treat the text as primarily English.
    return True


def compute_similarities(model, query: str, titles):
    if not titles:
        return []
    sentences = [query] + titles
    embeddings = model.encode(sentences, convert_to_tensor=True)
    query_emb = embeddings[0]
    title_embs = embeddings[1:]
    cos_scores = util.cos_sim(query_emb, title_embs)[0]
    return cos_scores.cpu().tolist()


def _run_searxng(query: str, count: int) -> List[Dict]:
    out: List[Dict] = []
    searx_results = search_searxng(
        query_url="http://127.0.0.1:8888/search",
        query=query,
        count=count,
        language="en",
        safesearch=0,
        categories="general",
    )
    for r in searx_results:
        title = (getattr(r, "title", "") or "").strip()
        link = (getattr(r, "link", "") or "").strip()
        if title and link:
            out.append({"title": title, "link": link, "engine": "searxng"})
    return out

def _run_mwmbl(query: str, count: int) -> List[Dict]:
    return search_mwmbl(query=query, count=count, timeout=20)


def get_all_search_content(query, count=100, sim_threshold=0.25, discard_ratio=0.3):
    """
    Search according to the query, first filter based on title similarity, then scrape the webpage and clean,
    Return a list of results sorted by similarity from large to small, and filtered by threshold and discard ratio.

    Args:
        query: Search keywords
        sim_threshold: Minimum similarity threshold, results below this value will be discarded
        discard_ratio: Discard ratio (0~1), e.g., 0.3 means approximately 30% of the lowest similarity results are expected to be discarded

    Returns:
        List[dict], each dict:
        {
            "title": str,
            "link": str,
            "content": str,
            "similarity": float,
        }
    """
    engine_tasks = {
        "searxng": lambda: _run_searxng(query, count),
        "mwmbl":  lambda: _run_mwmbl(query, count),
    }

    all_engine_items: List[Dict] = []
    max_engine_workers = min(len(engine_tasks), 8)

    engine_timeout = 10 
    with ThreadPoolExecutor(max_workers=max_engine_workers) as ex:
        future_to_engine = {ex.submit(fn): name for name, fn in engine_tasks.items()}
        for fut in as_completed(future_to_engine):
            eng = future_to_engine[fut]
            try:
                items = fut.result(timeout=engine_timeout)
                all_engine_items.extend(items)
            except Exception as e:
                print(f"[get_all_search_content] {eng} failed: {e}", flush=True)
    
# 4) TODO: translate comment
    dedup: Dict[str, Dict] = {}
    for item in all_engine_items:
        link = (item.get("link") or "").strip()
        title = (item.get("title") or "").strip()
        if not link or not title:
            continue
        if link not in dedup or len(title) > len(dedup[link].get("title", "")):
            dedup[link] = item
    merged_items = list(dedup.values())
    

    title_items = []
    for idx, result in enumerate(merged_items):
        link = (result.get("link") or "").strip()
        title = (result.get("title") or "").strip()
        if (
            not validate_language(title, link=link)
            or "pdf" in title.lower()
            or not validate_url(link, filter_url=['arxiv', 'academia.edu', 'juraforum.de', 'github', 'wiki/talk:', 'publication', 'youtube'])
        ):
            continue
        title_items.append({
            "title": title,
            "link": link,
            "engine": result.get("engine", "unknown"),
        })

    if not title_items:
        return []
    
    titles = [item["title"] for item in title_items]
    sims = compute_similarities(EMBEDDING_MODEL, query, titles)

    for item, sim in zip(title_items, sims):
        item["similarity"] = sim
    
    title_items.sort(key=lambda x: x["similarity"], reverse=True)

    discard_ratio = max(0.0, min(discard_ratio, 0.99))
    ratio_threshold = float("-inf")
    if 0 < discard_ratio < 1 and len(title_items) > 1:
        drop_n = int(len(title_items) * discard_ratio)
        keep_n = max(1, len(title_items) - drop_n)
        ratio_threshold = title_items[keep_n - 1]["similarity"]
    else:
        ratio_threshold = float("-inf")

    final_threshold = max(sim_threshold, ratio_threshold)

    title_filtered = [item for item in title_items if item["similarity"] >= final_threshold]
    title_filtered.sort(key=lambda x: x["similarity"], reverse=True)

    final_results = []
    max_workers = max(1, min(8, len(title_filtered)))
    def fetch_one(item):
        title = item["title"]
        link = item["link"]
        sim = item["similarity"]

        try:
            content = fetch_and_clean(link)
        except Exception:
            return None 
        if (
            not content
            or '403' in content
            or "[Request Failed]" in content
            or len(content.strip()) < 200
            or not validate_language(content, type="content", link=link)
        ):
            return None

        return {
            "title": title,
            "link": link,
            "content": content,
            "similarity": sim,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_item = {ex.submit(fetch_one, item): item for item in title_filtered}
        for fut in as_completed(future_to_item):
            try:
                result = fut.result()
            except Exception as e:
                print(f"[fetch_one] failed: {e}", flush=True)
                continue
            if result is not None:
                final_results.append(result)

    return final_results



def load_requests_grouped(req_files_by_level):
    """
    req_files_by_level: {level: [req_flag1, req_flag2, ...]}
    return:
      jobs_by_level: {level: [(job_id, req_flag, ready_flag, labels_list), ...]}
      merged_labels_by_level: {level: sorted_unique_labels}
    """
    jobs_by_level = defaultdict(list)
    merged_labels_by_level = defaultdict(list)

    for level, req_files in req_files_by_level.items():
        merged = []
        for req_flag in req_files:
            job_id = req_flag.split(f".L{level}.", 1)[1].split(".web_request", 1)[0]
            ready_flag = req_flag.replace(".web_request", ".web_ready")

            try:
                with open(req_flag, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                if not isinstance(labels, list):
                    continue
            except json.JSONDecodeError:
                continue

            jobs_by_level[level].append((job_id, req_flag, ready_flag, labels))
            merged.extend(labels)

        dedup = list(dict.fromkeys([x for x in merged if isinstance(x, str) and x.strip()]))
        merged_labels_by_level[level] = dedup

    return jobs_by_level, merged_labels_by_level

def write_ready_and_remove_requests(jobs, all_results):
    """
    jobs: [(job_id, req_flag, ready_flag, labels), ...]
    all_results: Dict[label] -> result_dict
    """
    for job_id, req_flag, ready_flag, labels in jobs:
        sub = {name: all_results.get(name, {
                    "wiki_titles": [],
                    "wiki_status": "no_result",
                    "wiki_valid": None,
                    "beta_param": (1, 1e7),
              })
              for name in labels}

        try:
            os.remove(req_flag)
        except FileNotFoundError:
            pass

        tmp_ready = ready_flag + ".tmp"
        with open(tmp_ready, "w", encoding="utf-8") as f:
            json.dump(sub, f, indent=2, ensure_ascii=False)
        os.replace(tmp_ready, ready_flag)
        print(f"[web_worker] Created ready: {ready_flag}", flush=True)
        
        
def process_one_level(level, labels, max_levels, existing_titles):
    if not labels:
        return {}, {}
    results, new_web_entries = process_level_with_web(labels, level, max_levels, existing_titles, count=200)
    return results, new_web_entries


MAX_REQ_FILES_PER_LEVEL_PER_ROUND = 50
LEVEL_PARALLELISM = 4 




def main():
    parser = argparse.ArgumentParser(description="web worker for KnowledgeTree.")
    parser.add_argument("--storage_path", type=str, required=True,
                        help="Directory path, used to save *_knowledge_tree.json / *_basic_tree.json and signal files")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name, used to concatenate file names")
    parser.add_argument("--specified_domain", type=str, required=True,
                        help="Top-level fixed domain, e.g. Mathematics")
    parser.add_argument("--max_levels", type=int, required=True,
                        help="KnowledgeTree maximum levels, used to traverse possible levels")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Polling interval for signal files (seconds)")
    parser.add_argument("--wiki_lang", type=str, default="en",
                        help="Wikipedia language code, e.g. 'en', 'zh', etc.")
    args = parser.parse_args()

    STORAGE_PATH = args.storage_path.rstrip("/")
    args.specified_domain = args.specified_domain.split(",")
    
    tag = "_".join(args.specified_domain)
    knowledge_tree_path = f"{STORAGE_PATH}/{args.model_name}_{tag}_knowledge_tree.json"
    basic_tree_path = f"{STORAGE_PATH}/{args.model_name}_{tag}_basic_tree.json"
    web_corpus_path = f"{STORAGE_PATH}/{args.model_name}_{tag}_web_corpus.json"
    
    end_flag_path = f"{basic_tree_path}.build_done"
    
    wikipedia.set_lang(args.wiki_lang)

    print("[web_worker] Started.")
    print(f"  storage_path      = {STORAGE_PATH}")
    print(f"  knowledge_tree    = {knowledge_tree_path}")
    print(f"  basic_tree        = {basic_tree_path}")
    print(f"  web_corpus_path    = {web_corpus_path}")
    print(f"  specified_domain  = {args.specified_domain}")
    print(f"  max_levels        = {args.max_levels}")
    print(f"  poll_interval     = {args.poll_interval} sec")
    print(f"  wiki_lang         = {args.wiki_lang}")
    print(f"  end_flag          = {end_flag_path}")

    
    while True:
        if os.path.exists(end_flag_path):
            print("[web_worker] Detected build_done, exit.")
            os.remove(end_flag_path)
            break

        req_files_by_level = {}
        for level in range(1, args.max_levels + 1):
            pattern = f"{basic_tree_path}.L{level}.*.web_request"
            req_files = sorted(glob.glob(pattern))
            if req_files:
                req_files_by_level[level] = req_files[:MAX_REQ_FILES_PER_LEVEL_PER_ROUND]

        if not req_files_by_level:
            time.sleep(args.poll_interval)
            continue

        jobs_by_level, merged_labels_by_level = load_requests_grouped(req_files_by_level)

        level_results = {}
        level_new_entries = {}
        corpus = load_web_corpus(web_corpus_path)
        existing_titles = set(corpus.keys())
        with ThreadPoolExecutor(max_workers=min(LEVEL_PARALLELISM, len(merged_labels_by_level))) as ex:
            futs = {}
            for level, labels in merged_labels_by_level.items():
                futs[ex.submit(process_one_level, level, labels, args.max_levels, existing_titles)] = level

            for fut in as_completed(futs):
                level = futs[fut]
                try:
                    res, newe = fut.result()
                    level_results[level] = res
                    level_new_entries[level] = newe
                except Exception as e:
                    print(f"[web_worker] Level {level} processing failed: {e}", flush=True)
                    level_results[level] = {}
                    level_new_entries[level] = {}
        
        merged_new = {}
        for e in level_new_entries.values():
            merged_new.update(e)

        if merged_new:
            added = 0
            for title, payload in merged_new.items():
                if title not in corpus:
                    corpus[title] = payload
                    added += 1
            if added > 0:
                save_web_corpus(web_corpus_path, corpus)
                
        for level, jobs in jobs_by_level.items():
            all_results = level_results.get(level, {})
            write_ready_and_remove_requests(jobs, all_results)

        time.sleep(0.1)


    
if __name__ == "__main__":
    main()
    
