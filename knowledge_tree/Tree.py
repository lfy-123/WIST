import json
import os
from threading import Lock
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Union
import uuid
import random
from collections import defaultdict
import torch
import numpy as np
from collections import OrderedDict
import re
from vllm import LLM, SamplingParams
import sys
import ray
from scipy.stats import chi2
import os
import time
from difflib import SequenceMatcher
from tqdm import tqdm
from .utils import load_web_corpus, save_web_corpus, show_path, split_path
from .prompts import build_expansion_prompt_math, build_expansion_prompt_med, build_expansion_prompt_phy
from typing import List, Dict, Any, Optional
import glob
import uuid, json, os, time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import concurrent.futures
# from .search_pipeline import search_and_fetch_grouped
import tempfile

@dataclass
class WebJob:
    job_id: str
    level: int
    explored_labels: List[str]
    req_flag: str
    ready_flag: str
    submit_time: float

def _normalize_label_for_similarity(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)  
    s = " ".join(s.split())            
    return s

def string_similarity(str1: str, str2: str) -> float:
    s1 = _normalize_label_for_similarity(str1)
    s2 = _normalize_label_for_similarity(str2)
    if not s1 or not s2:
        return 0.0
    base_sim = SequenceMatcher(None, s1, s2).ratio()
    len_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2))
    return base_sim * len_ratio


def best_wiki_match(label: str, wiki_keys: List[str]) -> Tuple[Optional[str], float]:
    best_title = None
    best_sim = 0.0
    for wiki_key in wiki_keys:
        sim = string_similarity(label, wiki_key)
        if sim > best_sim:
            best_sim = sim
            best_title = wiki_key
    return best_title, best_sim

# =========================
# Prompt TODO: translate comment
# =========================
SINGLE_CHUNK_CLEAN_PROMPT = """You are a strict cleaner for noisy web text. 
Your job is to REMOVE irrelevant/boilerplate parts and KEEP the useful content, 
while keeping the original meaning and level of detail.

Important (MUST follow):
- Do NOT summarize, do NOT explain, do NOT paraphrase.
- Do NOT add new information or examples.
- Do NOT compress multiple ideas into a shorter explanation.
- Only delete noise and fix formatting (line breaks / obvious duplication).

[Task]
Given raw text copied from a web page, produce a cleaned version:
- Keep all sentences that contain real content related to the page title.
- Remove obvious web noise.

Use the page title only to decide what is on-topic vs off-topic:
- page title: "{title}"

[REMOVE COMPLETELY – treat as noise]
Delete text that is clearly not the main content, such as:
- Navigation / UI: menus, buttons, "Prev", "Next", "Home", "Up", "Jump to content", search boxes, language selectors, breadcrumbs, etc.
- Site chrome and metadata: "From ...", "Updated on ...", revision history, copyright, emails, author bio, footer text.
- Login / sign up / subscribe / download / share / cookie notices / newsletter / ads / comments.
- Pure navigation lists: "Categories:", "Tags:", "External links:", long lists of links or language links.
- Long reference sections and citation lists: numbered [1], [2], DOIs, ISBNs, URLs, full bibliographies (you may delete them fully).
- Repeated page titles or headings that add no new content.
- Repeated boilerplate blocks that appear identical or almost identical.

[KEEP, but just clean formatting]
For sentences and paragraphs that are on-topic:
- Preserve the original meaning and technical detail.
- Keep formulas, symbols, step-by-step descriptions and code explanations.
- Merge broken lines and excessive newlines into normal paragraphs, 
  so that sentences are continuous and not split every few words.
- You may remove trivial repetitions of the same sentence/phrase.
- Do NOT rewrite the wording just to “sound better”.

[Style of output]
- Output should look like the same content, but with noise removed and line breaks cleaned.
- English only.
- No extra commentary or explanation.

[Output format]
Output ONLY the cleaned text, wrapped exactly as:
[Result Start]
...cleaned text...
[Result End]

Now clean the following raw web text:

[Raw web text]
{chunk_text}
"""


INCREMENTAL_CLEAN_PROMPT = """You are a strict cleaner for noisy web text. 
Your job is to MERGE and CLEAN text chunks from the same page by removing noise and formatting issues,
while preserving the original meaning and level of detail.

Important (MUST follow):
- Do NOT summarize, do NOT explain, do NOT paraphrase.
- Do NOT add new information.
- Do NOT compress different ideas into a shorter explanation.
- Only delete noise, deduplicate obvious repetitions, and fix formatting.

[Context]
You receive:
1) A previously cleaned text (from earlier chunks of the same page).
2) A new raw web text chunk from the same page.

Your goal:
- Combine them into a single cleaned version.
- Keep all useful, on-topic content from both.
- Remove off-topic noise and boilerplate.
- Remove duplicated sentences / paragraphs if they say the same thing.

Use the page title only to decide what is on-topic vs off-topic:
- page title: "{title}"

[REMOVE COMPLETELY – treat as noise]
Delete text that is clearly not the main content, such as:
- Navigation / UI: menus, buttons, "Prev", "Next", "Home", "Up", "Jump to content", etc.
- Site chrome and metadata: "From ...", "Updated on ...", revision history, copyright, emails, footer, author info.
- Login / sign up / subscribe / download / share / cookie / ads / comments.
- Categories/tags/external-links sections, link lists, language selectors.
- Long reference / bibliography / citation lists (numbered [1], [2], DOIs, URLs).
- Repeated page titles and other boilerplate text.

[KEEP & MERGE]
For on-topic sentences:
- Preserve original meaning and technical detail.
- Keep definitions, descriptions, formulas, steps, and important examples.
- Merge broken lines and excessive newlines into complete paragraphs.
- Remove exact or near-exact duplicate sentences/paragraphs.
- Keep the logical order roughly consistent with the original text:
  previous cleaned text first, then new information from the new chunk.

[Style of output]
- Output should look like the original content, but with noise removed, 
  duplicates reduced, and line breaks cleaned.
- Do NOT turn it into an abstract or explanation.

[Output format]
Output ONLY the updated cleaned text, wrapped exactly as:
[Result Start]
...merged cleaned text...
[Result End]

Now clean the following raw web text chunk and merge it with the existing cleaned text:

[Existing cleaned text]
<<<PREVIOUS_CLEANED_START>>>
{previous_summary}
<<<PREVIOUS_CLEANED_END>>>

[New raw web text chunk]
<<<NEW_CHUNK_START>>>
{chunk_text}
<<<NEW_CHUNK_END>>>
"""

class BasicNode:
    """Representation of a domain node in the tree"""
    def __init__(self, name: str, level: int, parent: 'BasicNode' = None, window_size: Union[int, str] = 'global'):
        self.name = name
        self.level = level
        self.parent = parent
        self.num_questions = 0
        self.children = OrderedDict()  # dict of nodes
        self.num_childs = 0
        if self.name == 'unk':
          self.is_unk = True
        else:
          self.is_unk = False

        if self.name == 'Root':
          self.beta_param = None
        else:
          self.beta_param = (1, 1)
        if window_size != 'global' and name != 'unk' and name != 'Root':
            self.window_size = int(window_size)
            self.window: List[int] = []  
        else:
            self.window_size = None
            self.window = None

        # ===== Wiki
        self.wiki_status: str = "unqueried"
        self.wiki_titles: List[str] = []
        self.wiki_best_title: Optional[str] = None
        self.wiki_valid: Optional[bool] = None
        self.is_valid: bool = True
        self.cleaned_corpus_titles = set()
        self.use_api = False
        self.web_search_results = []
        self.leaf_count = 0

    def add_child(self, node):
        """Add a child node with transition probability"""
        if node.name not in self.children:
            self.children[node.name] = node
            node.parent = self
            self.num_childs += 1

    def check_window(self):
        if self.window is None or self.window_size is None:
            return
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

    def update_beta_with_reward(self, reward: float) -> None:
        """
        reward > 0.4 -> alpha += 1 (Success = 1)
        reward < 0.4 -> beta += 1 (Failure = 0)
        reward == 0.4 -> No update
        """
        if self.name == "Root" or self.is_unk or self.beta_param is None:
            return
        if reward > 0.4:
            outcome = 1
        elif reward < 0.4:
            outcome = 0
        else:
            return
        a, b = self.beta_param
        if outcome == 1:
            a += 1
        else:
            b += 1
        self.beta_param = (int(a), int(b))
        if self.window is not None:
            self.window.append(outcome)
            self.check_window()
    
    def get_sampling_beta_param(self) -> Tuple[int, int]:
        if self.name == "Root" or self.beta_param is None:
            return (1, 1)
        if self.window_size is None or self.window is None:
            return (int(self.beta_param[0]), int(self.beta_param[1]))
        if len(self.window) == 0:
            return (1, 1)
        s = int(sum(self.window))
        n = int(len(self.window))
        return (1 + s, 1 + (n - s))
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "num_questions": self.num_questions,
            "window_size": self.window_size,
            "window": self.window,
            "beta_param": self.beta_param,
            "is_unk": self.is_unk,
            "children": [child.to_dict() for child in self.children.values()],
            "wiki_status": self.wiki_status,
            "wiki_titles": self.wiki_titles,
            "wiki_best_title": self.wiki_best_title,
            "wiki_valid": self.wiki_valid,
            "is_valid": self.is_valid,
            "use_api": self.use_api,
            "leaf_count": self.leaf_count,
            "web_search_results": self.web_search_results,
            "cleaned_corpus_titles": list(self.cleaned_corpus_titles),
        }

    @classmethod
    def from_dict(cls, data: dict, parent: 'BasicNode' = None) -> 'BasicNode':
        node = cls(data["name"], data["level"], parent)
        node.num_questions = data["num_questions"]
        node.beta_param = tuple(data["beta_param"]) if data["beta_param"] else None

        node.is_unk = data["is_unk"]
        node.window_size = data["window_size"]
        if node.window_size is not None:
            node.window = data["window"]
        else:
            node.window = None

        node.wiki_status = data.get("wiki_status", "unqueried")
        node.wiki_titles = data.get("wiki_titles", [])
        node.wiki_best_title = data.get("wiki_best_title")
        node.wiki_valid = data.get("wiki_valid", None)
        node.is_valid = data.get("is_valid", False)
        node.use_api = data.get("use_api", False)
        node.leaf_count = data.get("leaf_count", 0)
        
        node.web_search_results = data.get("web_search_results", [])
        node.cleaned_corpus_titles = set(data.get("cleaned_corpus_titles", []))
        node.num_childs = 0
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, parent=node)
            node.children[child.name] = child
            node.num_childs += 1
        return node

class BasicTree:
    def __init__(self, 
                max_levels: int = 10,
                knowledge_tree_path: str = "knowledge_tree.json",
                basic_tree_save_path: str = "basic_tree.json",
                basic_tree_load_path: str = None,
                window_size: Union[int, str] = 'global',
                thr: float = 0.1,
                explore_strategy: str = 'reward',
                fixed_domains: List = None,
                fixed_domains_ratio: List[int] = None,
                seed: int = 42,
                reward_update_version = "v1",
                web_corpus_path=None,
                max_nodes_nums_once: dict = {"1":10, "2":10, "3": 10, "4": 20},
                times_for_nodes_limit = 10,
                wait_interval: int = 10,
                tokenizer = None,
                clean_prompt_max_len = 16384,
                clean_max_new_tokens = 16384,
                corpus_chunk_tokens = 8192,
                corpus_overlap_tokens = 1024,
                clean_web: bool = False,
                prompt_version: str = "v1",
                specified_backbone: dict = None,
                expand_tree_or_not: bool = True,
                select_top_k_nodes: int = 1,
                min_count: int = 2,
                disable_thinking_mode_or_not = None
                ) -> None:
        self.max_levels = max_levels
        self.knowledge_tree_path = knowledge_tree_path
        self.basic_tree_save_path = basic_tree_save_path
        self.window_size = window_size  
        self.thr = thr
        self.lock = Lock()
        self.explore_strategy = explore_strategy
        self.fixed_domains = fixed_domains
        self.specified_backbone = specified_backbone
        self.reward_update_version = reward_update_version
        
        self.max_nodes_nums_once = max_nodes_nums_once
        self.times_for_nodes_limit = times_for_nodes_limit
        self.wait_interval = wait_interval
        self.tokenizer = tokenizer
        self.clean_prompt_max_len = clean_prompt_max_len
        
        self.corpus_chunk_tokens = corpus_chunk_tokens
        self.corpus_overlap_tokens = corpus_overlap_tokens
        self.clean_max_new_tokens = clean_max_new_tokens
        self.clean_web = clean_web
        self.basic_tree_load_path = basic_tree_load_path
        self.prompt_version = prompt_version
        self.select_top_k_nodes = select_top_k_nodes
        self.min_count = min_count
        self.disable_thinking_mode_or_not = disable_thinking_mode_or_not

        if fixed_domains_ratio is not None:
            assert len(fixed_domains_ratio) == len(fixed_domains), "fixed_domains_ratio length must match fixed_domains length."
        if basic_tree_load_path is None or not os.path.exists(basic_tree_load_path):
            self._initialize_file()
            self.root = BasicNode("Root", level=0)
            self.levels = {i: [] for i in range(self.max_levels + 1)}
            self.levels[0] = ['Root']
            self.levels[1] = self.fixed_domains
            for idx, domain in enumerate(self.fixed_domains):
                node_window_size = self.window_size if self.window_size is not None else 'global'
                node = BasicNode(domain, level=1, window_size=node_window_size)
                if fixed_domains_ratio is not None:
                    node.beta_param = (int(fixed_domains_ratio[idx]), 1)
                self.root.add_child(node)
                if expand_tree_or_not:  
                    unk_node = BasicNode("unk", level=2)
                    node.add_child(unk_node)
                if self.specified_backbone and domain in self.specified_backbone:
                    print(f"[BasicTree] Building specified backbone subtree for domain '{domain}' and still_add_unk={expand_tree_or_not}")
                    self._build_backbone_subtree(
                        parent_node=node,
                        subtree=self.specified_backbone[domain],
                        current_level=2,
                        still_add_unk=expand_tree_or_not,
                    )
            with self.lock:
                with open(self.knowledge_tree_path, 'r') as f:
                    dict_tree = json.load(f)
                if self.specified_backbone is not None:
                    dict_tree["Root"] = {}
                    for domain in self.fixed_domains:
                        if domain in self.specified_backbone:
                            dict_tree["Root"][domain] = self.specified_backbone[domain]
                        else:
                            dict_tree["Root"][domain] = {}
                else:
                    dict_tree["Root"] = {domain: {} for domain in self.fixed_domains}
                temp_file = f"{self.knowledge_tree_path}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(dict_tree, f, indent=2)
                os.replace(temp_file, self.knowledge_tree_path)
            self.node_counter = {i: len(self.levels[i]) for i in range(len(self.levels))}
        else:
            self._load_from_file()
            if not expand_tree_or_not:  
                unk_nodes = self.find_unk_nodes()
                for unk in unk_nodes:
                    unk.beta_param = (1, 1e7)

        self.web_corpus_path = web_corpus_path
        self.web_corpus = {}  
        self.corpus_items = set()  
        self.is_or_not_cleaned = False  
        self.update_web_corpus()
        self.all_nodes_name_set = set([domain.strip().lower() for domain in self.get_all_domains()])
        self.web_executor = ThreadPoolExecutor(max_workers=8)
        self.sampled_paths = []     
        self.sampled_paths_str = [] 
        self._last_save_ts = 0.0    
        self.pending_web_jobs: List[WebJob] = []
        self.web_job_nodes: Dict[str, List[BasicNode]] = {}
        self._set_all_windows(self.root)  
        self._recover_web_ready_files_strict(delete_after_import=True)

    def set_rng_state(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    
    def _recover_web_ready_files_strict(self, delete_after_import: bool = True) -> int:
        abs_tree_path = os.path.abspath(self.basic_tree_save_path)
        tree_dir = os.path.dirname(abs_tree_path)
        tree_base = os.path.basename(abs_tree_path)
        rx = re.compile(
            re.escape(tree_base) + r"\.L(?P<lvl>\d+)\.(?P<job>[0-9a-fA-F]{12})\.web_ready$"
        )
        try:
            names = os.listdir(tree_dir)
        except FileNotFoundError:
            return 0
        ready_files = []
        for fn in names:
            if rx.match(fn):
                ready_files.append(os.path.join(tree_dir, fn))
        ready_files.sort()
        print(f"[recover_web_ready] dir={tree_dir}, matched_ready={len(ready_files)}")
        if not ready_files:
            return 0
        imported_files = []
        any_update = False
        for rf in ready_files:
            m = rx.match(os.path.basename(rf))
            if not m:
                continue
            level = int(m.group("lvl"))
            jobid = m.group("job")
            if level < 1 or level > self.max_levels:
                continue
            name2nodes = defaultdict(list)
            for node in self._iter_nodes_at_level(level, specified_domain=None):
                name2nodes[node.name].append(node)
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    web_results = json.load(f)
            except Exception as e:
                print(f"[recover_web_ready] skip unreadable: L{level}.{jobid} err={e}")
                continue
            if not isinstance(web_results, dict):
                print(f"[recover_web_ready] skip non-dict json: L{level}.{jobid}")
                continue
            file_updated = False
            miss = 0
            for node_name, result in web_results.items():
                if not isinstance(result, dict):
                    continue
                nodes = name2nodes.get(node_name, [])
                if not nodes:
                    miss += 1
                    continue
                wiki_titles = result.get("wiki_titles", [])
                if wiki_titles is None:
                    wiki_titles = []
                if not isinstance(wiki_titles, list):
                    wiki_titles = [wiki_titles]
                wiki_status = result.get("wiki_status", None)
                wiki_valid = result.get("wiki_valid", None)
                beta_param = result.get("beta_param", None)
                beta_tuple = None
                if isinstance(beta_param, (list, tuple)) and len(beta_param) == 2:
                    try:
                        beta_tuple = (int(beta_param[0]), int(beta_param[1]))
                    except Exception:
                        beta_tuple = None
                for node in nodes:
                    node.wiki_titles = list(set(wiki_titles))
                    if wiki_status is not None:
                        node.wiki_status = wiki_status
                    if "wiki_valid" in result:
                        node.wiki_valid = wiki_valid
                    if beta_tuple is not None:
                        node.beta_param = beta_tuple
                file_updated = True
                any_update = True
            if file_updated:
                imported_files.append(rf)
                print(f"[recover_web_ready] imported: {os.path.basename(rf)} miss_keys={miss}")
            else:

                print(f"[recover_web_ready] no match, keep: {os.path.basename(rf)} keys={len(web_results)} miss={miss}")
        if not imported_files:
            return 0
        try:
            self.update_web_corpus()
            self.save_to_file()
        except Exception as e:
            print(f"[recover_web_ready] save failed, will NOT delete ready files. err={e}")
            return 0
        deleted = 0
        if delete_after_import:
            for rf in imported_files:
                try:
                    if os.path.exists(rf):
                        os.remove(rf)
                except Exception as e:
                    print(f"[recover_web_ready] delete ready failed: {rf} err={e}")
                    continue
                reqf = rf.replace(".web_ready", ".web_request")
                try:
                    if os.path.exists(reqf):
                        os.remove(reqf)
                except Exception:
                    pass
                deleted += 1
        print(f"[recover_web_ready] imported & deleted {deleted} ready files.")
        return deleted

    def _set_all_windows(self, current_node):
        if self.window_size != 'global' and current_node.name != 'unk' and current_node.name != 'Root':
            if current_node.window_size is not None:
                return
            current_node.window_size = int(self.window_size)
            current_node.window: List[int] = []  
        for child in current_node.children.values():
            self._set_all_windows(child)

    def _build_backbone_subtree(self, parent_node: 'BasicNode', subtree: dict, current_level: int, still_add_unk: bool = True):
        if current_level > self.max_levels:
            return
        if current_level not in self.levels:
            self.levels[current_level] = []
        for name, child_subtree in subtree.items():
            node = BasicNode(name, level=current_level)
            parent_node.add_child(node)
            if current_level + 1 <= self.max_levels and still_add_unk:
                unk_child = BasicNode("unk", level=current_level + 1)
                node.add_child(unk_child)
            self.levels[current_level].append(name)
            if isinstance(child_subtree, dict) and len(child_subtree) > 0 and current_level + 1 <= self.max_levels:
                self._build_backbone_subtree(
                    parent_node=node,
                    subtree=child_subtree,
                    current_level=current_level + 1,
                )

    def update_web_corpus(self):
        if not self.web_corpus_path or not os.path.exists(self.web_corpus_path):
            return
        web_entries = load_web_corpus(self.web_corpus_path)
        assert isinstance(web_entries, dict), "Web corpus file must contain dict."
        for title, item in web_entries.items():
            content = item.get("content", "").strip()
            similarity = item.get("similarity", None)
            
            cleaned_content = item.get("cleaned_content", "").strip()
            if title in self.corpus_items and self.web_corpus[title][1] == cleaned_content:
                continue
            
            if similarity is not None and similarity < 0.5:
                self.web_corpus[title] = ("", "")
                self.corpus_items.add(title)
                continue
            
            assert len(content) > 0, f"Web corpus title '{title}' has empty content."
            if title and content:
                self.web_corpus[title] = (content, cleaned_content)
                self.corpus_items.add(title)
        print(f"[update_web_corpus] Loaded {len(web_entries)} entries from web corpus.")
    
    def get_nodes_by_path(self, path, sep="->"):
        if isinstance(path, str):
            parts = [
                p.strip().strip('"').strip("'")
                for p in path.split(sep)
                if p.strip().strip('"').strip("'")
            ]
        elif isinstance(path, (list, tuple)):
            parts = [str(p).strip().strip('"').strip("'") for p in path if str(p).strip()]
        else:
            raise TypeError(f"Unsupported path type: {type(path)}. Use str or list of str.")
        nodes = [self.root]
        if not parts:
            return nodes
        current = self.root
        idx = 0
        if hasattr(current, "name") and parts[0] == current.name:
            nodes.append(current)
            idx = 1
        for name in parts[idx:]:
            if not hasattr(current, "children") or name not in current.children:
                return []
            current = current.children[name]
            nodes.append(current)
        return nodes
        
    def _initialize_file(self):
        """Initialize the knowledge tree structure as json"""
        with self.lock:
            if not os.path.exists(self.knowledge_tree_path):
                os.makedirs(os.path.dirname(self.knowledge_tree_path), exist_ok=True)
                with open(self.knowledge_tree_path, 'w') as f:
                    json.dump({}, f)

    def save_to_file(self):
        data = {
            "max_levels": self.max_levels,
            "root": self.root.to_dict(),
            "levels": self.levels,
            "node_counter": self.node_counter
        }
        with self.lock:
            temp_file = f"{self.basic_tree_save_path}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.basic_tree_save_path)
        time.sleep(0.1)  

    def _load_from_file(self):
        print(f"Load label tree from the path {self.basic_tree_load_path}")
        with self.lock:
            with open(self.basic_tree_load_path, 'r') as f:
                data = json.load(f)
        data["levels"] = {int(k): v for k, v in data["levels"].items()}
        data["node_counter"] = {int(k): v for k, v in data["node_counter"].items()}
        self.max_levels = data["max_levels"]
        self.root = BasicNode.from_dict(data["root"]) 
        self.levels = data["levels"]
        self.node_counter = data["node_counter"]
        self._rebuild_parent_links(self.root)
    
    def _rebuild_parent_links(self, node: BasicNode, parent: BasicNode = None):
        node.parent = parent
        for child in node.children.values():
            self._rebuild_parent_links(child, parent=node)

    def get_all_domains(self, specified_domain: str = None):
        if specified_domain:
            if specified_domain not in self.root.children:
                raise ValueError(f"Specified domain '{specified_domain}' not found in level 1 domains")
            flat_list = []
            def collect_domains(node: BasicNode):
                if node.name != 'unk':
                    flat_list.append(node.name)
                for child in node.children.values():
                    collect_domains(child)
            collect_domains(self.root.children[specified_domain])
            return flat_list
        else:
            flat_list = []
            for level, domains in self.levels.items():
                flat_list.extend(domains)
            return flat_list

    def build_path_to_node(self, node: BasicNode) -> List[BasicNode]:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def find_unk_nodes(self, specified_domain: str = None) -> List['BasicNode']:
        unk_nodes = []
        def collect_unk_nodes(node: BasicNode):
            if node.name == 'unk':
                unk_nodes.append(node)
            for child in node.children.values():
                collect_unk_nodes(child)
        if specified_domain:
            if specified_domain not in self.root.children:
                raise ValueError(f"Specified domain '{specified_domain}' not found in level 1 domains")
            collect_unk_nodes(self.root.children[specified_domain])
        else:
            collect_unk_nodes(self.root)
        return unk_nodes
    
    def _sample_child_by_beta_param(self, children):
        if self.explore_strategy == "reward":
            alphas = np.array([c.get_sampling_beta_param()[0] for c in children], dtype=float)
            betas  = np.array([c.get_sampling_beta_param()[1] for c in children], dtype=float)
            samples = np.random.beta(alphas, betas)
            return children[int(np.argmax(samples))]
        else:
            weights = np.array([getattr(c, "leaf_count", 1) for c in children], dtype=np.float64)

            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
            if weights.sum() <= 0:

                idx = np.random.randint(len(children))
            else:
                p = weights / weights.sum()
                idx = np.random.choice(len(children), p=p)
            return children[int(idx)]
    
    def _sample_path_async(
        self,
        models: List,
        specified_domain: str = None,
        only_sample: bool = False,
        max_retry: int = 5,      
        max_steps: int = 2000,     
    ):
        path = [self.root]
        current_node = self.root

        if specified_domain:
            if specified_domain not in self.root.children:
                raise ValueError(f"Specified domain '{specified_domain}' not found")
            current_node = self.root.children[specified_domain]
            path.append(current_node)

        step_cnt = 0
        retry_cnt = 0
        reset_cnt = 0

        def reset_to_root():
            nonlocal path, current_node, retry_cnt, reset_cnt
            path = [self.root]
            current_node = self.root
            if specified_domain:
                current_node = self.root.children[specified_domain]
                path.append(current_node)
            retry_cnt = 0
            reset_cnt += 1
            print(f"[sample_async] reset to root, reset_cnt={reset_cnt}")

        while current_node.level < self.max_levels:
            step_cnt += 1

            if step_cnt >= max_steps or retry_cnt >= max_retry:
                reset_to_root()
                continue

            if only_sample:
                children = [
                    c for k, c in current_node.children.items()
                    if k != "unk" and c.beta_param[1] < 1e6
                ]
            else:
                children = [child for key, child in current_node.children.items() if child.beta_param[1] < 1e6]  #  if child.beta_param[1] < 1e6

            if not children:
                retry_cnt += 1
                reset_to_root()
                continue
            next_node = self._sample_child_by_beta_param(children)

            if next_node.name == "unk":
                explored_labels, new_nodes = self.explore_node(models, next_node)
                if not explored_labels:
                    retry_cnt += 1
                    continue
                job = self.submit_web_request(
                    level=next_node.level,
                    explored_labels=explored_labels,
                )
                self.pending_web_jobs.append(job)
                self.web_job_nodes[job.job_id] = new_nodes
                retry_cnt = 0
                continue
            path.append(next_node)
            current_node = next_node
            retry_cnt = 0
        return path
    
    def drop_both(self, key: str):
        pre_length = len(self.sampled_paths)
        keep = []
        keep_str = []
        for p, s in zip(self.sampled_paths, self.sampled_paths_str):
            if key not in split_path(s):
                keep.append(p)
                keep_str.append(s)
        self.sampled_paths = keep
        self.sampled_paths_str = keep_str
        return pre_length - len(self.sampled_paths)
    
    def _drain_ready_web_jobs(
        self,
        models: List,
        specified_domain: str = None,
        sampled_paths: Optional[List[str]] = None,
        max_jobs: int = 20,
        job_timeout_sec: int = 1800,
    ):
        if not self.pending_web_jobs:
            return
        sampled_paths = sampled_paths if sampled_paths is not None else []
        done_jobs = []
        now = time.time()
        for job in self.pending_web_jobs:
            if len(done_jobs) >= max_jobs:
                break
            if os.path.exists(job.ready_flag):
                done_jobs.append(job)
        if not done_jobs:
            return
        for job in done_jobs:
            pass
        new_pending = []
        for job in self.pending_web_jobs:
            if job in done_jobs:
                continue
            new_pending.append(job)
        self.pending_web_jobs = new_pending

        for job in done_jobs:
            if not os.path.exists(job.ready_flag):
                self.web_job_nodes.pop(job.job_id, None)
                continue
            try:
                with open(job.ready_flag, "r", encoding="utf-8") as f:
                    web_results = json.load(f)
            except Exception as e:
                print(f"[drain] failed to read ready_flag: {job.ready_flag}, err={e}")
                self.web_job_nodes.pop(job.job_id, None)
                continue
            try:
                os.remove(job.ready_flag)
            except Exception:
                pass
            new_nodes = self.web_job_nodes.pop(job.job_id, [])
            for node in new_nodes:
                result_dict = (web_results or {}).get(node.name, {})
                node.wiki_titles = list(set(result_dict.get("wiki_titles", [])))
                node.wiki_status = result_dict.get("wiki_status", "no_result")
                node.wiki_valid = result_dict.get("wiki_valid", None)
                node.beta_param = result_dict.get("beta_param", (1, 1e7))
            self.update_web_corpus()
            if job.level < self.max_levels:
                self.validate_level_with_llm(
                    target_level=job.level,
                    models=models,
                    specified_domain=specified_domain,
                    sampled_paths=sampled_paths,   
                )
            self.save_to_file()
    
    def _split_text_into_chunks_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 128,
    ) -> List[str]:
        """
        Tokenize the text by token length and then decode it back into a string.
        Overlap the blocks by the number of tokens.
        """
        if not text:
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        all_ids = encoded["input_ids"]
        n_tokens = len(all_ids)

        if n_tokens <= max_tokens + overlap_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < n_tokens:
            end = min(n_tokens, start + max_tokens)
            sub_ids = all_ids[start:end]
            chunk_text = self.tokenizer.decode(sub_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end >= n_tokens:
                break

            start = max(0, end - overlap_tokens)
        return chunks

    def _extract_result_block(self, raw_text: str) -> str:
        """
        Extract the content between [Result Start] ... [Result End] using regular expressions.
        If not found, return the original text after stripping (as a fallback).
        """
        if not raw_text:
            return ""
        RESULT_BLOCK_RE = re.compile(r"\[Result Start\](.*?)\[Result End\]", re.S)
        m = RESULT_BLOCK_RE.search(raw_text)
        if m:
            result = m.group(1).strip()
            if "merged cleaned text" in result:
                return ""
            return result
        print("[clean_web_corpus_with_llm] WARNING: '[Result Start]/[Result End]' not found, using raw text as fallback.")
        return raw_text.strip()
    
    def disable_thinking_mode(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        return text
        
    
    def _run_vllm_multi_engines_single_prompt(
        self,
        vllm_engines: List,
        prompt: str,
        sampling_params: SamplingParams,
    ):
        if not vllm_engines:
            raise ValueError("The vllm_engines variable is empty.")

        if self.disable_thinking_mode_or_not is not None:
            prompt = self.disable_thinking_mode(prompt)
        
        tok = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.clean_prompt_max_len,
            return_tensors=None,
        )
        input_ids = tok["input_ids"]
        if isinstance(input_ids[0], list):
            prompt_token_ids = input_ids[0]
        else:
            prompt_token_ids = input_ids
        refs = []
        for eng in vllm_engines:
            refs.append(
                eng.add_requests.remote(
                    sampling_params=sampling_params,
                    prompt_token_ids=[prompt_token_ids],  
                )
            )
        ray.get(refs)
        output_refs = [eng.get_responses.remote() for eng in vllm_engines]
        all_outputs = sum(ray.get(output_refs), [])
        return all_outputs
    
    def _run_vllm_multi_engines_batch_prompts(
        self,
        vllm_engines: List,
        prompts: List[str],
        sampling_params: SamplingParams,
    ):
        if not vllm_engines:
            raise ValueError("The vllm_engines variable is empty.")
        if not prompts:
            return []

        if self.disable_thinking_mode_or_not is not None:
            prompts = [self.disable_thinking_mode(p) for p in prompts]

        batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            max_length=self.clean_prompt_max_len,
            truncation=True,
        )
        all_prompt_token_ids = batch["input_ids"]
        num_prompts = len(all_prompt_token_ids)

        add_refs = []
        for eng in vllm_engines:
            add_refs.append(
                eng.add_requests.remote(
                    sampling_params=sampling_params,
                    prompt_token_ids=all_prompt_token_ids,
                )
            )
        ray.get(add_refs)

        get_refs = [eng.get_responses.remote() for eng in vllm_engines]
        all_engine_outputs = ray.get(get_refs)
        grouped_outputs = [[] for _ in range(num_prompts)]
        for engine_outputs in all_engine_outputs:
            if len(engine_outputs) != num_prompts:
                print(
                    f"[WARN] engine returned {len(engine_outputs)} outputs, "
                    f"expected {num_prompts}"
                )
            for i, out in enumerate(engine_outputs):
                if i < num_prompts:
                    grouped_outputs[i].append(out)

        return grouped_outputs


    def _run_vllm_batch(
        self,
        vllm_engines: List[Any],
        prompts: List[str],
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        if not prompts:
            return []

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            max_tokens=max_new_tokens,
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            logprobs=None,
            stop=stop,
        )

        if self.disable_thinking_mode_or_not is not None:
            new_prompts = []
            for prompt in prompts:
                new_prompts.append(self.disable_thinking_mode(prompt))
            prompts = new_prompts
        
        batch = self.tokenizer(
            prompts,
            add_special_tokens=False,
            max_length=self.clean_prompt_max_len,
            truncation=True,
        )
        all_prompt_token_ids: List[List[int]] = batch["input_ids"]

        llms = vllm_engines
        num_llms = len(llms)
        num_prompts = len(prompts)


        batch_size = (num_prompts + num_llms - 1) // num_llms

        add_refs = []
        index_ranges = []  
        for i, llm in enumerate(llms):
            start = i * batch_size
            end = min(num_prompts, (i + 1) * batch_size)
            if start >= end:

                index_ranges.append((start, start))
                continue
            sub_token_ids = all_prompt_token_ids[start:end]
            index_ranges.append((start, end))
            add_refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params,
                    prompt_token_ids=sub_token_ids,
                )
            )
        if add_refs:
            ray.get(add_refs)

        output_texts = [""] * num_prompts
        get_refs = []
        for llm in llms:
            get_refs.append(llm.get_responses.remote())

        all_engine_outputs: List[List[Any]] = ray.get(get_refs)

        for (start, end), engine_outputs in zip(index_ranges, all_engine_outputs):
            if start >= end:
                continue
            if len(engine_outputs) != (end - start):
                print(
                    f"[_run_vllm_batch] WARNING: engine returned {len(engine_outputs)} outputs, "
                    f"but expected {end - start} (start={start}, end={end})"
                )
            for offset, out in enumerate(engine_outputs):
                idx = start + offset
                if idx >= num_prompts:
                    break
                if not out.outputs:
                    output_texts[idx] = ""
                else:
                    output_texts[idx] = out.outputs[0].text
        return output_texts

    def clean_web_corpus_with_llm(
        self,
        target_level: int,
        need_cleaned_names: List[str],
        models: List,  
        specified_domain: Optional[str] = None,
        max_source_tokens: int = 2048,
        chunk_overlap_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ):
        vllm_engines = models  
        web_entries = load_web_corpus(self.web_corpus_path)
        nodes = self._iter_nodes_at_level(target_level, specified_domain=specified_domain)
        print(f"[clean_web_corpus_with_llm] level={target_level} has {len(nodes)} nodes (not all need cleaning)")
        jobs: List[Dict[str, Any]] = []
        for node in nodes:
            if need_cleaned_names and node.name not in need_cleaned_names:
                continue
            web_titles = node.wiki_titles or []
            if not web_titles:
                print(f"[clean_web_corpus_with_llm] node '{node.name}' web has no corpus info, skip cleaning")
                continue
            for title in web_titles:
                title = title.strip()
                if not title:
                    continue
                if getattr(node, "cleaned_corpus_titles", None) and title in node.cleaned_corpus_titles:
                    print(f"[clean_web_corpus_with_llm] node '{node.name}''s corpus title '{title}' has been cleaned, skipped")
                    continue
                raw_tuple = self.web_corpus.get(title, ("", ""))
                content = raw_tuple[0]
                if not content:
                    print(
                        f"Warning!!! [clean_web_corpus_with_llm] node '{node.name}' has an empty corpus title '{title}', skipping cleaning"
                    )
                    raise ValueError(f"Web corpus title '{title}' has no content.")
                chunks = self._split_text_into_chunks_by_tokens(
                    content,
                    max_tokens=max_source_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
                if not chunks:
                    print(
                        f"[clean_web_corpus_with_llm] node '{node.name}''s corpus title '{title}' is empty after segmentation, skipped"
                    )
                    continue
                jobs.append(
                    {
                        "node": node,
                        "title": title,
                        "content": content,
                        "chunks": chunks,
                        "cleaned": "",  
                    }
                )
        if not jobs:
            print("[clean_web_corpus_with_llm] No corpus needs to be cleaned, returning directly")
            return
        print(f"[clean_web_corpus_with_llm] Collected {len(jobs)} documents to clean")
        max_chunks = max(len(job["chunks"]) for job in jobs)
        for chunk_idx in range(max_chunks):
            round_prompts: List[str] = []
            round_job_indices: List[int] = []
            for job_idx, job in enumerate(jobs):
                if chunk_idx >= len(job["chunks"]):
                    continue
                chunk_text = job["chunks"][chunk_idx]
                node = job["node"]
                title = job["title"]

                if chunk_idx == 0 and not job["cleaned"]:

                    prompt = SINGLE_CHUNK_CLEAN_PROMPT.format(
                        title=title,
                        chunk_text=chunk_text,
                    )
                else:

                    prompt = INCREMENTAL_CLEAN_PROMPT.format(
                        title=title,
                        previous_summary=job["cleaned"],
                        chunk_text=chunk_text,
                    )
                encoded = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                all_ids = encoded["input_ids"]
                n_tokens = len(all_ids)
                if n_tokens > self.clean_prompt_max_len:
                    print(
                        f"[clean_web_corpus_with_llm] WARNING: prompt construction is too long."
                        f"node='{node.name}', title='{title}', chunk_idx={chunk_idx}, "
                        f"prompt tokens={n_tokens} > max {self.clean_prompt_max_len}, skipped cleaning"
                    )
                    continue
                round_prompts.append(prompt)
                round_job_indices.append(job_idx)

            if not round_prompts:
                continue
            print(
                f"[clean_web_corpus_with_llm] chunk round {chunk_idx + 1}/{max_chunks}, "
                f"this round needs to clean {len(round_prompts)} chunks"
            )

            raw_outputs = self._run_vllm_batch(
                vllm_engines=vllm_engines,
                prompts=round_prompts,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stop=["[Result End]"],
            )

            for local_idx, raw_text in enumerate(raw_outputs):
                job_idx = round_job_indices[local_idx]
                job = jobs[job_idx]

                cleaned_segment = self._extract_result_block(raw_text)
                if not cleaned_segment:
                    print(
                        f"[clean_web_corpus_with_llm] WARNING: extracted result is empty, fallback to original chunk, "
                        f"node='{job['node'].name}', title='{job['title']}'"
                    )
                    print(f"model cleaned raw output was:\n{raw_text}")
                    if job["cleaned"]:
                        cleaned_segment = job["cleaned"] + "\n" + job["chunks"][chunk_idx]
                    else:
                        cleaned_segment = job["chunks"][chunk_idx]
                job["cleaned"] = cleaned_segment.strip()

        for job in jobs:
            node = job["node"]
            title = job["title"]
            original_content = job["content"]
            cleaned_content = (job["cleaned"] or original_content).strip()
            self.web_corpus[title] = (original_content, cleaned_content)
            if title not in web_entries:
                web_entries[title] = {}
            web_entries[title]["cleaned_content"] = cleaned_content
            node.cleaned_corpus_titles.add(title)

        save_web_corpus(self.web_corpus_path, web_entries)
        print("[clean_web_corpus_with_llm] All cleaned results have been written back to web_corpus_path")


    def explore_node(self, models: List, unk: BasicNode):
        """Explore and expand the tree at the specified 'unk' node"""
        path = self.build_path_to_node(unk)
        parent_node = path[-2]
        print(f"[explore_node] ======================= Expanding: {show_path(path[:-1])}'s sub-domains =======================")
        num_invalid_try = 0
        sub_labels = []
        exist_domains = [key for key, child in path[-2].children.items() if key != "unk"]
        
        sibling_domains_of_parent = [key for key, child in path[-3].children.items() if key != "unk"] if path[-3].name != "Root" else []  #  and child.beta_param[1] < 1e6
        
        num_exist_domains = len(exist_domains)
        while num_invalid_try < 3:
            proposed_new_domain_list, model_responses = self.sprout_with_llm(path[:-1], exist_domains, models, sibling_domains_of_parent, top_k=self.select_top_k_nodes)  # root -> ... -> node_n
            
            if proposed_new_domain_list is None:
                print("[explore_node] Failed to generate a valid new domain, counted as one invalid attempt.")
                num_invalid_try += 1
                continue
            
            pre_nums = len(sub_labels)
            for proposed_new_domain in proposed_new_domain_list:
                if proposed_new_domain is not None and "*" in proposed_new_domain:
                    proposed_new_domain = proposed_new_domain.replace("*", "").strip()

                if (proposed_new_domain is None or proposed_new_domain == 'None' 
                    or not self._validate_proposal(proposed_new_domain) 
                    or proposed_new_domain.lower() == parent_node.name.lower()):
                    print(f"[explore_node] proposed_new_domain={proposed_new_domain}, failed to generate a valid new domain.")
                    continue
                
                if proposed_new_domain in exist_domains:
                    print(f"[explore_node] proposed_new_domain={proposed_new_domain}, already exists.")
                    continue
                
                if proposed_new_domain.strip().lower() in self.all_nodes_name_set:
                    print(f"[explore_node] proposed_new_domain={proposed_new_domain}, already exists in the whole tree.")
                    continue
                
                sub_labels.append(proposed_new_domain)
                exist_domains.append(proposed_new_domain)
                if len(sub_labels) >= self.max_nodes_nums_once[str(unk.level)]:  
                    print(f"[explore_node] Reached the upper limit of sub-domains {self.max_nodes_nums_once[str(unk.level)]}, expansion stopped.")
                    sub_labels = sub_labels[:self.max_nodes_nums_once[str(unk.level)]]
                    num_invalid_try += 100  
                    break
            if len(sub_labels) == pre_nums:
                print("[explore_node] No valid new domain generated in this attempt, counted as one invalid attempt.")
                num_invalid_try += 1
        if len(sub_labels) == 0:
            print(f"[explore_node] No valid new domain generated, expansion terminated.")
            return None, None
        
        cleaned_sub_labels = sub_labels
        
        final_sub_labels = []
        final_new_nodes = []
        for proposed_new_domain in cleaned_sub_labels:
            if not self._validate_proposal(proposed_new_domain):
                continue
            if self.window_size is not None:
                new_node = BasicNode(proposed_new_domain, level=unk.level, window_size = self.window_size)
            else:
                new_node = BasicNode(proposed_new_domain, level=unk.level)
            final_sub_labels.append(proposed_new_domain)
            final_new_nodes.append(new_node)
            
            if new_node.level < self.max_levels:
                new_node.add_child(BasicNode('unk', level=new_node.level + 1))
            
            parent_node.add_child(new_node)  
            self.levels[new_node.level].append(new_node.name)
            self.all_nodes_name_set.add(new_node.name.strip().lower())
            
            self.node_counter[new_node.level] += 1
            
            with self.lock:
                with open(self.knowledge_tree_path, 'r') as f:
                    dict_tree = json.load(f)
                current = dict_tree["Root"]
                for i in range(1, len(path) - 1):
                    current = current[path[i].name]
                current[new_node.name] = {}
                temp_file = f"{self.knowledge_tree_path}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(dict_tree, f, indent=2)
                os.replace(temp_file, self.knowledge_tree_path)
        print(f"[explore_node] Expansion completed. Under the path {show_path(path[:-1])}, there were {num_exist_domains} labels, and now {len(final_sub_labels)} new domains have been generated: {', '.join(final_sub_labels)}")

        if num_exist_domains + len(final_sub_labels) >= self.max_nodes_nums_once[str(parent_node.level)] * self.times_for_nodes_limit:
            print(f"[explore_node] Due to the upper limit of the number of sub-domains, the unk node under the node {show_path(path[:-1])} will be masked.")
            unk.beta_param = (1, 1e7)
        return final_sub_labels, final_new_nodes
    
    def sample_paths(
        self,
        batch_size: int,
        models: List,
        only_sample: bool = False,
        specified_domain: str = None,
        paths_ckpt_path: str = None,
        sampled_paths=None,
        drain_every: int = 2,     
        drain_max_jobs: int = 50,  
        drain_interval_sec: float = 10,      
        drain_time_budget_sec: float = 600,   
    ):
        if sampled_paths is not None:
            self.sampled_paths_str = list(sampled_paths)
            print(f"[sample_paths] Sampled paths have been passed from the outside, a total of {len(self.sampled_paths_str)} paths.")
            self.sampled_paths = []
            for path in sampled_paths:
                node_path = self.get_nodes_by_path(path)
                if (not node_path) or (len(node_path) != self.max_levels + 1):
                    print(f"[sample_paths] drop invalid ckpt path: {path}")
                    continue
                self.sampled_paths.append(node_path)
        else:
            self.sampled_paths_str = []
            self.sampled_paths = []
        last_drain = 0.0
        while len(self.sampled_paths) < batch_size:
            path = self._sample_path_async(models, specified_domain=specified_domain, only_sample=only_sample)
            if not path:
                continue
            self.sampled_paths.append(path)
            self.sampled_paths_str.append(show_path(path))
            if paths_ckpt_path:
                with open(paths_ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(self.sampled_paths_str, f, indent=2, ensure_ascii=False)
            need_drain = (len(self.sampled_paths) % drain_every == 0)
            now = time.time()
            if (now - last_drain) >= drain_interval_sec:
                need_drain = True
            if need_drain:
                self._drain_ready_web_jobs(
                    models=models,
                    specified_domain=specified_domain,
                    sampled_paths=self.sampled_paths_str,  
                    max_jobs=drain_max_jobs,
                )
                last_drain = time.time()
            print(f"[sample_paths] kept={len(self.sampled_paths)}/{batch_size}, pending_web={len(self.pending_web_jobs)}")
        for _ in range(5):
            self._drain_ready_web_jobs(models=models, max_jobs=drain_max_jobs)
            if len(self.sampled_paths) >= batch_size:
                break
            time.sleep(0.1)
        while len(self.sampled_paths) < batch_size:
            path = self._sample_path_async(models, specified_domain=specified_domain, only_sample=only_sample)
            self.sampled_paths.append(path)
            self.sampled_paths_str.append(show_path(path))
        return self.sampled_paths[:batch_size]

    def _normalize_label(self, label: str) -> str:
        """Label normalization for comparison/deduplication: remove extra spaces + all lowercase"""
        return re.sub(r'\s+', ' ', label).strip().lower()

    def sprout_with_llm(self, path: List, exist_domains: List, models: List, sibling_domains: List[str] = [], top_k: int = 1) -> Optional['BasicNode']:       
        current_level = path[-1].level
        child_level = current_level + 1
        is_leaf_level = (child_level == self.max_levels)
        if self.prompt_version == "math":
            prompt = build_expansion_prompt_math(path, exist_domains, is_leaf_level, sibling_domains)
        elif self.prompt_version == "med":
            prompt = build_expansion_prompt_med(path, exist_domains, is_leaf_level, sibling_domains)
        elif self.prompt_version == "phy":
            prompt = build_expansion_prompt_phy(path, exist_domains, is_leaf_level, sibling_domains)
        else:
            raise ValueError(f"Unknown prompt_version: {self.prompt_version}. Expected one of: math, med, phy")
        
        
        vllm_engines = models
        temp_random = random.randint(5, 12) * 0.1
        
        sampling_params = SamplingParams(
            temperature=temp_random,
            top_p=0.95,
            max_tokens=4096,
            n=5,  
            stop=["[Proposition End]"],
            include_stop_str_in_output=True,
        )


        all_outputs = self._run_vllm_multi_engines_single_prompt(
            vllm_engines=vllm_engines,
            prompt=prompt,
            sampling_params=sampling_params,
        )
        proposals = []
        model_responses = []
        for req_out in all_outputs:
            for out in req_out.outputs:
                raw_text = (out.text or "")
                model_responses.append(raw_text)
                match = re.search(
                    r'\[Proposition Start\](.+?)\[Proposition End\]',
                    raw_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if not match:
                    continue
                proposal = match.group(1).strip()
                proposal = re.sub(r'\s+', ' ', proposal)
                if self._validate_proposal(proposal):
                    proposals.append(proposal)
        
        if not proposals:
            return None, model_responses
        
        normalized_exist = {self._normalize_label(d) for d in exist_domains}
        
        from collections import defaultdict
        counts = defaultdict(int)
        norm2canonical = {}
        for p in proposals:
            norm_p = self._normalize_label(p)
            if norm_p in normalized_exist:
                continue
            counts[norm_p] += 1

            if norm_p not in norm2canonical:
                norm2canonical[norm_p] = p
        
        if not counts:
            return None, model_responses
        repeated = [(norm, c) for norm, c in counts.items() if c >= self.min_count]
        print(f"[sprout_with_llm] Found {len(repeated)} unique proposals with count >= {self.min_count}")
        
        if not repeated:
            return None, model_responses
        repeated.sort(key=lambda x: x[1], reverse=True)
        selected_norms = [norm for norm, _ in repeated[:max(top_k, 1)]]
        selected_labels = [norm2canonical[norm] for norm in selected_norms]
        
        cleaned_labels = [
            lbl for lbl in selected_labels
            if "no more" not in lbl.lower().strip()
        ]
        if not cleaned_labels:
            return None, model_responses
        kind = "knowledge point" if is_leaf_level else "sub-domain"
        print(
            f"Proposed new {kind}(s) on level {child_level}: "
            + ", ".join(cleaned_labels)
        )
        return cleaned_labels, model_responses

    def _validate_proposal(self, proposal: str) -> bool:
        """Validate a proposed domain name with Title Case normalization"""
        if not proposal:
            return False
        proposal = proposal.strip()
        _STOPWORD_TOKENS = {
            "and", "or", "the", "a", "an", "if", "then", "else", "for", "to", "of", "<name>", "description", "name:"
        }
        tokens = proposal.lower().split()
        if len(tokens) == 1:
            t = tokens[0]
            if t in _STOPWORD_TOKENS or len(t) <= 2:
                return False

        if len(proposal) == 0 or len(proposal) > 100:
            return False
        if re.search(r'[\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]', proposal):
            print(f"[Verification Failed] proposal contains non-English CJK/Korean/Arabic characters: {proposal}")
            return False
        ALLOWED_PUNCT_ASCII = set("-_',:()^*/+=.·\\[]{}|<>~")
        ALLOWED_EXTRA_RANGES = [
            (0x2070, 0x209F),  # superscript/subscript
            (0x2200, 0x22FF),  # math symbols
            (0x0391, 0x03A9),  # Greek uppercase
            (0x03B1, 0x03C9),  # Greek lowercase
        ]

        ALLOWED_EXTRA_CHARS = {
            "\u2032",  # ′ prime
            "–",       # U+2013 en dash
            "—",       # U+2014 em dash
        }
        for ch in proposal:

            if ch.isspace():
                continue
            code = ord(ch)
            if ch.isascii():
                if ch.isalnum() or ch in ALLOWED_PUNCT_ASCII:
                    continue
                print(f"[Verification Failed] proposal contains non-English CJK/Korean/Arabic characters: {repr(ch)} in {proposal}")
                return False
            if ch.isalpha():
                continue
            if ch in ALLOWED_EXTRA_CHARS:
                continue
            if any(start <= code <= end for (start, end) in ALLOWED_EXTRA_RANGES):
                continue
            print(f"[Verification Failed] proposal contains non-English CJK/Korean/Arabic characters: {repr(ch)} in {proposal}")
            return False
        lower = proposal.lower()

        forbidden_substrings = [
            "proposed",   
            "proposal",   
            "completed", 
            "assistant",  
            "user",       
            "system",     
            "next level", 
            "proof",
            "corrected",
            "confirmed",
            "label",
            "12x8",       
            "description",
            "name",
        ]
        if any(s in lower for s in forbidden_substrings):
            return False
        if re.search(r"[.;!\?\[\]\{\}]", proposal):
            return False
        tokens = re.findall(r"[a-z0-9']+", lower)
        forbidden_words = {"unk", "unknown", "none", "maybe"}
        if any(tok in forbidden_words for tok in tokens):
            return False
        return True

    def _nodes_from_path_str(self, path_str: str) -> Optional[List[BasicNode]]:
        if not path_str or not isinstance(path_str, str):
            return None
        parts = re.split(r"\s*(?:->|→|/|\|)\s*", path_str.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        if parts[0].lower() == "root":
            parts = parts[1:]
        node = self.root
        nodes: List[BasicNode] = []
        for name in parts:
            if name not in node.children:
                return None
            node = node.children[name]
            nodes.append(node)
        return nodes

    def update_with_reward_feedback(self, path_to_max_reward: Dict[str, List[float]], save_bool = True) -> Dict[str, Any]:
        """
        Input:
        `path_to_max_reward: {path_str: [max_r1, max_r2, ...], ...}`

        Behavior:
        For each path's reward, perform the following along all nodes:
        - If reward > 0.4, increment alpha by 1.
        - If reward < 0.4, increment beta by 1.
        - If reward == 0.4, skip.

        Window:
        If a node has `window_size` enabled, record the outcome (0/1) in the window (automatically truncated).

        Return:
        Basic statistics (number of paths updated, number of feedbacks, etc.).
        """
        if not path_to_max_reward:
            return {"updated_paths": 0, "updated_events": 0, "skipped_paths": 0}
        updated_paths = 0
        updated_events = 0
        skipped_paths = 0

        with self.lock:
            for path_str, rewards in path_to_max_reward.items():
                nodes = self._nodes_from_path_str(path_str)
                if nodes is None:
                    skipped_paths += 1
                    continue
                if not isinstance(rewards, list):
                    rewards = [rewards]
                any_event_on_this_path = False
                for r in rewards:
                    try:
                        rr = float(r)
                    except Exception:
                        continue
                    for node in nodes:
                        node.update_beta_with_reward(rr)
                    updated_events += 1
                    any_event_on_this_path = True
                if any_event_on_this_path:
                    updated_paths += 1
        if save_bool:
            self.save_to_file()
        return {"updated_paths": updated_paths, "updated_events": updated_events, "skipped_paths": skipped_paths}
    
    def _build_children_filter_prompt(self,
                                  path: List['BasicNode'],
                                  parent_name: str,
                                  all_children: List[str],
                                  is_leaf_level: bool) -> str:
        """
        Construct a prompt for "unified cleaning of all subdomains of the same parent node":
        all_children = all sub-labels determined in this round (old + new), and the model can only select a subset of them.
        """
        path_str = show_path(path)
        children_str = "\n".join(f"- {c}" for c in all_children)

        if not is_leaf_level:
            level_desc = "subdomains (mathematical subfields)"
        else:
            level_desc = "atomic knowledge points (single theorems / definitions / standard constructions)"

        prompt = f"""You are cleaning the children of a node in a hierarchical mathematics knowledge tree.

Current path (from root to this parent):
{path_str}

Parent node:
"{parent_name}"

The current children at this level (candidates to keep or discard) are:
{children_str}

Your task:
- Decide which of these labels should remain as direct children of "{parent_name}" at THIS level.
- All remaining children must be {level_desc} with a similar level of generality.

You MUST REMOVE a label if:
- It is essentially a duplicate or almost-duplicate of another label in the list.
- It is clearly a more specific topic that belongs under another label in the list, not as a sibling. For example:
    "K-theory" and "K-theory in Algebraic Topology"
    "Algebraic K-theory" and "Algebraic K-theory of symmetric groups"
    "Spectral Sequences" and "Spectral Sequences in Algebraic Topology"
- It just adds obvious qualifiers like "in Algebraic Topology", "in Topology", "Applications of X to Y"
  when there is already a more basic label in the list.
- It is from a different mathematical area than the path suggests.
- It is vague, non-standard, or clearly not a usual mathematical topic at this level.

Important constraints:
- You may ONLY keep or discard labels from the list above.
- You MUST NOT invent any new labels.
- You MUST NOT rename or rephrase any label.
- The final output must be a subset of the input labels, with each kept label written EXACTLY as it appears above.

Output format (very important):
- Enclose your final list between [Result Start] and [Result End].
- Inside the markers, output ONE kept label per line, exactly copied from the list above.
- If you think NO label should be kept, leave the area empty (just place [Result Start] and [Result End] on consecutive lines).

Example:

[Result Start]
Algebraic Topology
Homotopy Theory
K-theory
[Result End]

Now output ONLY the [Result Start] ... [Result End] block."""
        return prompt


    def _cleanup_children_with_llm(self,
                               parent_node: 'BasicNode',
                               exist_domains: List[str],
                               sub_labels: List[str],
                               models: List) -> List[str]:
        """
        Use the model to clean up all the sub-labels under the "all sub-labels list", but **only return the retained part of the newly generated labels sub_labels in this round**.

        Parameters:
        - parent_node: The current parent node
        - exist_domains: The current "all sub-labels list", which should include: existing sub-labels + newly generated sub_labels in this round
        - sub_labels: The candidate sub-labels just generated in this round (not yet actually attached to the tree)
        - models: LLM list, default to the first one for cleaning

        Returns:
        - cleaned_sub_labels: The list of new labels that are still retained (a subset of sub_labels)
        """
        if not sub_labels:
            return []
        print(f"="*30)
        print(f"Starting to use the model to clean up all the sub-labels under the node '{parent_node.name}': {exist_domains}")
        print(f"="*30)

        if not exist_domains:
            exist_domains = sub_labels[:]

        path = self.build_path_to_node(parent_node)
        is_leaf_level = (parent_node.level + 1 == self.max_levels)
        all_children = exist_domains
        prompt = self._build_children_filter_prompt(
            path=path,
            parent_name=parent_node.name,
            all_children=all_children,
            is_leaf_level=is_leaf_level
        )
        
        vllm_engines = models
        temp_random = random.randint(5, 12) * 0.1
        sampling_params = SamplingParams(
            temperature=temp_random,
            top_p=0.95,
            max_tokens=1024,
            n=3,
            stop=["[Result End]"],
            include_stop_str_in_output=True,
        )
        all_outputs = self._run_vllm_multi_engines_single_prompt(
            vllm_engines=vllm_engines,
            prompt=prompt,
            sampling_params=sampling_params,
        )

        for req_out in all_outputs:
            for out in req_out.outputs:
                raw_text = (out.text or "")
                m = re.search(
                    r'\[Result Start\](.+?)\[Result End\]',
                    raw_text,
                    re.DOTALL | re.IGNORECASE,
                )
                if not m:
                    continue
                body = m.group(1).strip()
                if not "\n" in body or "..." in body:
                    print(f"[cleanup_children_with_llm] The model may not have listed the labels in rows or made a mistake, skipping this output: {body}")
                    continue
                returned_labels_raw = []
                if body:
                    for line in body.splitlines():
                        label = line.strip().strip("-*•").strip()
                        if label:
                            returned_labels_raw.append(label)

                lower2orig = {}
                for lbl in all_children:
                    lower = lbl.lower()
                    lower2orig.setdefault(lower, lbl)

                kept_all = []
                kept_lower = set()
                for lbl in returned_labels_raw:
                    lower = lbl.lower()
                    if lower in lower2orig:
                        orig = lower2orig[lower]
                        if orig not in kept_all:
                            kept_all.append(orig)
                            kept_lower.add(lower)
                    else:
                        print(f"[cleanup_children_with_llm] The model returned a label '{lbl}' that is not in exist_domains, which has been ignored.")

                if not kept_all:
                    continue

                sub_lower_set = {s.lower() for s in sub_labels}
                cleaned_sub_labels = []
                seen_lower = set()
                for lbl in kept_all:
                    lower = lbl.lower()
                    if lower in sub_lower_set and lower not in seen_lower:
                        if hasattr(self, "_validate_proposal") and not self._validate_proposal(lbl):
                            print(f"[cleanup_children_with_llm] '{lbl}' did not pass the _validate_proposal check, skipping.")
                            continue
                        cleaned_sub_labels.append(lbl)
                        seen_lower.add(lower)

                if not cleaned_sub_labels:
                    print(
                        f"[cleanup_children_with_llm] The original sub_labels has {len(sub_labels)} labels, "
                        f"and {len(cleaned_sub_labels)} labels are retained after cleaning: {cleaned_sub_labels}"
                    )
                    continue

                print(
                    f"[cleanup_children_with_llm] The original sub_labels has {len(sub_labels)} labels, "
                    f"and {len(cleaned_sub_labels)} labels are retained after cleaning: {cleaned_sub_labels}"
                )
                return cleaned_sub_labels
        print(f"[cleanup_children_with_llm] The model was cleaned and no labels were retained, reverting to the original. sub_labels。")
        return sub_labels
        
        
    def _request_flag(self, level: int, flag = "wiki") -> str:
        return f"{self.basic_tree_save_path}.L{level}.{flag}_request"

    def _ready_flag(self, level: int, flag = "wiki") -> str:
        return f"{self.basic_tree_save_path}.L{level}.{flag}_ready"
    
    
    def submit_web_request(self, level: int, explored_labels: List[str]) -> WebJob:
        """
        Create a request file with a unique job_id, and return a WebJob (which will poll the ready_flag by the drain function).
        """
        job_id = uuid.uuid4().hex[:12]
        req_flag = f"{self.basic_tree_save_path}.L{level}.{job_id}.web_request"
        ready_flag = f"{self.basic_tree_save_path}.L{level}.{job_id}.web_ready"
        tmp = req_flag + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(explored_labels, f, indent=2, ensure_ascii=False)
        os.replace(tmp, req_flag)
        return WebJob(
            job_id=job_id,
            level=level,
            explored_labels=explored_labels,
            req_flag=req_flag,
            ready_flag=ready_flag,
            submit_time=time.time(),
        )
    
    def wait_web_ready(self, level: int, interval: int = 60, explored_labels: List[str] = []):
        """
        After generating a certain layer:
        1. Create a web_request marker file to let the external web program know to process this layer;
        2. Check the web_ready marker file every interval seconds;
        3. Once web_ready appears, delete it and reload basic_tree.
        """
        req_flag = self._request_flag(level, flag="web")
        ready_flag = self._ready_flag(level, flag="web")

        if os.path.exists(req_flag):
            os.remove(req_flag)
        if os.path.exists(ready_flag):
            os.remove(ready_flag)
        
        with open(req_flag, "w") as f:
            json.dump(explored_labels, f, indent=2)
        print(f"[wait_web_ready] Created web request file: {req_flag}")
        
        sleep_turns = 0
        while True:
            if os.path.exists(ready_flag):
                print(f"[wait_web_ready] Detected web completed file: {ready_flag}")
                with open(ready_flag, "r") as f:
                    web_results = json.load(f)
                os.remove(ready_flag)
                break
            if sleep_turns % 10 == 0:
                print(f"[wait_web_ready] Waiting for web processing level={level} ... Already waited {interval * sleep_turns} seconds ... Checking file: {ready_flag}")
            time.sleep(interval)
            sleep_turns += 1
        return web_results
    
    def _iter_nodes_at_level(self, level: int, specified_domain: Optional[str] = None):
        """DFS traversal, return all nodes at the specified level (non-unk)"""
        result = []
        if specified_domain:
            start_nodes = [self.root.children[specified_domain]] if specified_domain in self.root.children else []
        else:
            start_nodes = list(self.root.children.values())

        stack = start_nodes[:]
        while stack:
            node = stack.pop()
            if node.level == level and not node.is_unk:
                result.append(node)
            for child in node.children.values():
                stack.append(child)
        return result

    def validate_level_with_llm(
        self,
        target_level: int,
        models: List,
        specified_domain: Optional[str] = None,
        sampled_paths=None
    ):
        sampled_paths = sampled_paths if sampled_paths is not None else []
        nodes = self._iter_nodes_at_level(target_level, specified_domain=specified_domain)
        print(f"[validate_level_with_llm(batch+multi-engine)] level={target_level}, nodes={len(nodes)}")
        is_leaf_level = (target_level == self.max_levels)
        vllm_engines = models

        llm_nodes = []
        llm_prompts = []

        for node in nodes:
            if not node.is_valid:
                continue

            wiki_keys = node.wiki_titles or []
            if not wiki_keys:
                node.wiki_status = "invalid"
                node.wiki_valid = False
                node.is_valid = False
                node.beta_param = (1, 1e7)
                dropped = self.drop_both(node.name)
                if dropped:
                    print(f"[validate] dropped {dropped} sampled_paths containing '{node.name}'")
                continue

            best_title, best_sim = best_wiki_match(node.name, wiki_keys)
            if best_title is not None and best_sim >= 0.80:
                node.wiki_status = "validated"
                node.wiki_valid = True
                node.wiki_best_title = best_title
                continue

            path = self.build_path_to_node(node)
            prompt = self._build_check_prompt(path, node.name, wiki_keys, is_leaf_level)
            llm_nodes.append(node)
            llm_prompts.append(prompt)

        if not llm_prompts:
            return

        temp_random = random.randint(5, 12) * 0.1
        sampling_params = SamplingParams(
            temperature=temp_random,
            top_p=0.95,
            max_tokens=512,
            n=1,  # engine TODO: translate comment
            stop=["[Proposition End]"],
            include_stop_str_in_output=True,
        )

        grouped_outputs = self._run_vllm_multi_engines_batch_prompts(
            vllm_engines=vllm_engines,
            prompts=llm_prompts,
            sampling_params=sampling_params,
        )

        for node, req_outputs in zip(llm_nodes, grouped_outputs):
            candidates = []

            for req_out in req_outputs:
                for out in req_out.outputs:
                    raw_text = out.text or ""
                    m = re.search(
                        r"\[Proposition Start\](.+?)\[Proposition End\]",
                        raw_text,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if m:
                        candidates.append(m.group(1).strip())

            if not candidates:
                node.wiki_status = "invalid"
                node.wiki_valid = False
                node.is_valid = False
                node.beta_param = (1, 1e7)
                dropped = self.drop_both(node.name)
                if dropped:
                    print(f"[validate] dropped {dropped} sampled_paths containing '{node.name}'")
                continue

            counter = defaultdict(int)
            for c in candidates:
                counter[c] += 1
            final_choice, _ = max(counter.items(), key=lambda x: x[1])

            if final_choice.lower() == "wrong":
                node.wiki_status = "invalid"
                node.wiki_valid = False
                node.is_valid = False
                node.beta_param = (1, 1e7)
                dropped = self.drop_both(node.name)
                if dropped:
                    print(f"[validate] dropped {dropped} sampled_paths containing '{node.name}'")
            else:
                new_name = final_choice.strip()
                if not self._validate_proposal(new_name):
                    node.wiki_status = "invalid"
                    node.wiki_valid = False
                    node.is_valid = False
                    node.beta_param = (1, 1e7)
                    dropped = self.drop_both(node.name)
                    if dropped:
                        print(f"[validate] dropped {dropped} sampled_paths containing '{node.name}'")
                else:
                    node.wiki_status = "validated"
                    node.wiki_valid = True
                    node.wiki_best_title = new_name


