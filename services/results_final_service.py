"""
Final results normalizer (early-return, no try/except)

Rules (mapping):
 - pipeline 'definite_exam_answer'   -> FAST
 - pipeline 'definite_retrieve_answer' -> DEFINITE
 - pipeline 'suggest_top3_bm25'      -> POSSIBLE
 - pipeline 'no_bm25_result'         -> UNKNOW
 - any error/unexpected              -> FAIL
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json, pathlib, re, logging
from enum import Enum
from difflib import get_close_matches, SequenceMatcher

from services import get_results_bm25_and_semantic_service, load_data_semantic_mock
from services.compare_score_multi_exam_service import compare_score_multi_exam_service

# =========================
# Enum & regex
# =========================
class Status(Enum):
    DEFINITE = 1
    POSSIBLE = 2
    UNKNOW = 3
    FAST = 4
    FAIL = 5

_RE_WORDS = re.compile(r"[A-Za-zÀ-ỹ0-9_]+")

# =========================
# Helpers: normalize & paths
# =========================
# Use central normalize_text from preprocess_data_service to keep behaviour consistent
from services.preprocess_data_service import normalize_text as _norm_text

def _repo_paths() -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    # Walk upwards from this file to locate the repository root.
    # Prefer a parent that contains a `data` folder, `.git` or `pyproject.toml`.
    p = pathlib.Path(__file__).resolve()
    repo_root = None
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() or (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    if repo_root is None:
        repo_root = p.parent

    lesson_dir = repo_root / "data" / "lesson"
    dialogue_dir = repo_root / "data" / "dialogue"
    return repo_root, lesson_dir, dialogue_dir

# =========================
# Subject map (id -> title)
# =========================
_SUBJECT_MAP: dict = {}

def _load_subject_map() -> dict:
    global _SUBJECT_MAP
    if _SUBJECT_MAP:
        return _SUBJECT_MAP

    repo_root, _, _ = _repo_paths()
    p = repo_root / "data" / "sub.json"
    if not p.exists():
        _SUBJECT_MAP = {}
        return _SUBJECT_MAP

    arr = json.loads(p.read_text(encoding="utf-8"))
    temp: dict = {}
    for it in arr if isinstance(arr, list) else [arr]:
        sid = (it or {}).get("_id") or (it or {}).get("id")
        title = (it or {}).get("tittle") or (it or {}).get("title")
        if sid is None or not title:
            continue
        key = int(sid) if (isinstance(sid, int) or (isinstance(sid, str) and sid.isdigit())) else str(sid)
        temp[key] = str(title)
    _SUBJECT_MAP = temp
    return _SUBJECT_MAP

# =========================
# Build "source" string
# =========================
def _build_source_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    """Build a human-readable source such as "{subject_name} - Bài {lesson}".

    Important corrections:
    - Do NOT treat meta['file'] as a lesson id. Many records carry filename strings there.
    - Try lesson only from 'lesson' or 'lesson_id'. If absent, try to extract a numeric
      lesson id from 'lesson_file'/'file' like "lesson_12.json".
    - If we can't determine a reliable subject or lesson, return None (avoid wrong defaults).
    """
    if not meta or not isinstance(meta, dict):
        return None

    def _extract_lesson_id(m: Dict[str, Any]) -> Optional[str]:
        val = m.get("lesson") or m.get("lesson_id")
        if isinstance(val, (int, str)) and str(val).strip():
            return str(val).strip()
        # Try to parse numeric id from file-like fields
        for k in ("lesson_file", "file", "file_name"):
            s = m.get(k)
            if isinstance(s, str) and s.strip():
                # find last integer group in the string
                mobj = re.search(r"(\d+)", s)
                if mobj:
                    return str(int(mobj.group(1)))
        return None

    subj = meta.get("subject") or meta.get("subject_id")
    lesson_part = _extract_lesson_id(meta)

    subj_name: Optional[str] = None
    if subj is not None:
        if isinstance(subj, int) or (isinstance(subj, str) and str(subj).isdigit()):
            smap = _load_subject_map()
            # Subject id -> name, or keep id as string as last resort
            subj_name = smap.get(int(subj)) or str(subj)
        elif isinstance(subj, str) and subj.strip():
            subj_name = subj.strip()

    # If we have a lesson id but no subject name, try infer subject by scanning lesson/dialogue
    if lesson_part and not subj_name:
        inferred = _find_subject_by_lesson(lesson_part)
        subj_name = inferred  # may remain None if not found

    # Build final label cautiously
    if subj_name and lesson_part:
        return f"{subj_name} - Bài {lesson_part}"
    if subj_name:
        return subj_name
    if lesson_part:
        # Avoid claiming a generic subject when we only know lesson id
        return f"Bài {lesson_part}"
    return None

def _find_subject_by_lesson(lesson_part: str) -> Optional[str]:
    _, lesson_dir, dialogue_dir = _repo_paths()

    for folder in (lesson_dir, dialogue_dir):
        if not folder.exists():
            continue
        for p in folder.glob("*.json"):
            arr = json.loads(p.read_text(encoding="utf-8"))
            items = arr if isinstance(arr, list) else [arr]
            for item in items:
                lv = (item or {}).get("lesson") or (item or {}).get("lesson_id") or (item or {}).get("file")
                if lv is None:
                    continue
                lv_str = str(lv).strip()
                if (lv_str.isdigit() and str(int(lv_str)) == str(lesson_part).strip()) or (lv_str == str(lesson_part).strip()):
                    subj = (item or {}).get("subject") or (item or {}).get("subject_id") or (item or {}).get("subjectId")
                    title = (item or {}).get("tittle") or (item or {}).get("title")
                    if subj is None:
                        return str(title) if title else None
                    if isinstance(subj, int) or (isinstance(subj, str) and str(subj).isdigit()):
                        smap = _load_subject_map()
                        return smap.get(int(subj)) or (str(title) if title else str(subj))
                    return str(subj)
    return None

# =========================
# Map suggestion -> semantic source
# =========================
def _map_suggestion_to_semantic_source(
    sugg: Dict[str, Any], sem_records: List[Dict[str, Any]], id_index: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    if not sugg:
        return None

    rid = sugg.get("rid") or sugg.get("doc_id") or sugg.get("id")
    if rid is not None and id_index:
        key = int(rid) if (isinstance(rid, int) or (isinstance(rid, str) and rid.isdigit())) else str(rid)
        rec = id_index.get(key)
        if rec:
            src = _build_source_from_meta(rec.get("meta") if isinstance(rec, dict) else None)
            if src:
                return src

    ans_text = (sugg.get("ans") or sugg.get("answer") or sugg.get("text") or "").strip()
    if ans_text:
        n_ans = _norm_text(ans_text)
        for r in sem_records:
            txt = (r.get("text") or "").strip()
            if n_ans and _norm_text(txt).find(n_ans) != -1:
                src = _build_source_from_meta(r.get("meta") if isinstance(r.get("meta"), dict) else None)
                if src:
                    return src

    pool = [r.get("text") or "" for r in sem_records if r.get("text")]
    if pool:
        close = get_close_matches(ans_text, pool, n=1, cutoff=0.5)
        if close:
            target = close[0]
            for r in sem_records:
                if r.get("text") == target:
                    src = _build_source_from_meta(r.get("meta") if isinstance(r.get("meta"), dict) else None)
                    if src:
                        return src

        # fallback: best ratio
        best, best_ratio = None, 0.0
        for r in sem_records:
            txt = r.get("text") or ""
            ratio = SequenceMatcher(None, ans_text, txt).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, r
        if best is not None and best_ratio >= 0.6:
            src = _build_source_from_meta(best.get("meta") if isinstance(best.get("meta"), dict) else None)
            if src:
                return src
    return None

# =========================
# Resolve source via lesson meta (semantic)
# =========================
_LESSON_CACHE: dict = {}

def _load_json_cached(folder: pathlib.Path, filename: str) -> Optional[List[Dict[str, Any]]]:
    if filename in _LESSON_CACHE:
        return _LESSON_CACHE[filename]
    p = folder / str(filename)
    if not p.exists():
        _LESSON_CACHE[filename] = None
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    _LESSON_CACHE[filename] = obj if isinstance(obj, list) else None
    return _LESSON_CACHE[filename]

def _resolve_source_from_lesson_meta(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not meta or not isinstance(meta, dict):
        return None

    # Only accept lesson id from dedicated fields; never directly from 'file' unless numeric is extractable
    lesson_id_raw = meta.get("lesson") or meta.get("lesson_id")
    lesson_file = meta.get("lesson_file") or meta.get("file_name") or meta.get("file")

    # Normalize lesson id: allow numeric extraction from filenames if explicit id missing
    if lesson_id_raw is None and isinstance(lesson_file, str):
        mobj = re.search(r"(\d+)", lesson_file)
        lesson_id_raw = str(int(mobj.group(1))) if mobj else None

    if lesson_file is None and isinstance(lesson_id, str) and lesson_id.endswith(".json"):
        lesson_file, lesson_id = lesson_id, None

    if lesson_file is None and lesson_id is None:
        return None

    l_id: Any
    if isinstance(lesson_id, int):
        l_id = lesson_id
    elif isinstance(lesson_id, str) and lesson_id.isdigit():
        l_id = int(lesson_id)
    else:
        l_id = lesson_id

    _, lesson_dir, dialogue_dir = _repo_paths()

    lf: Optional[List[Dict[str, Any]]] = None
    if lesson_file:
        lf = _load_json_cached(lesson_dir, str(lesson_file))

    if lf is None and l_id is not None:
        # scan lesson dir
        for p in lesson_dir.glob("*.json"):
            key = str(p.name)
            arr = _load_json_cached(lesson_dir, key)
            if not isinstance(arr, list):
                continue
            for item in arr:
                lv = (item or {}).get("lesson") or (item or {}).get("lesson_id")
                if lv is None:
                    continue
                if (isinstance(lv, int) and isinstance(l_id, int) and lv == l_id) or (
                    isinstance(lv, str) and lv.isdigit() and int(lv) == int(l_id)  # type: ignore[arg-type]
                ):
                    lf, lesson_file = arr, key
                    break
            if lf is not None:
                break

    if lf is None and l_id is not None:
        # scan dialogue dir
        for p in dialogue_dir.glob("*.json"):
            key = str(p.name)
            arr = _load_json_cached(dialogue_dir, key)
            if not isinstance(arr, list):
                continue
            for item in arr:
                lv = (item or {}).get("lesson") or (item or {}).get("lesson_id")
                if lv is None:
                    continue
                if (isinstance(lv, int) and isinstance(l_id, int) and lv == l_id) or (
                    isinstance(lv, str) and lv.isdigit() and int(lv) == int(l_id)  # type: ignore[arg-type]
                ):
                    lf, lesson_file = arr, key
                    break
            if lf is not None:
                break

    if not isinstance(lf, list):
        return None

    # build source
    for item in lf:
        lv = (item or {}).get("lesson") or (item or {}).get("lesson_id")
        if l_id is not None and lv is not None:
            same = (lv == l_id) or (
                isinstance(lv, int) and isinstance(l_id, int) and lv == l_id
            ) or (isinstance(lv, str) and lv.isdigit() and int(lv) == int(l_id))  # type: ignore[arg-type]
            if not same:
                continue

        subj = (item or {}).get("subject") or (item or {}).get("subject_id") or (item or {}).get("subjectId")
        title = (item or {}).get("tittle") or (item or {}).get("title")
        lesson_label = lv if lv is not None else l_id

        if subj is None:
            return f"{title} - Bài {lesson_label}" if title and lesson_label is not None else None

        if isinstance(subj, int) or (isinstance(subj, str) and str(subj).isdigit()):
            smap = _load_subject_map()
            subj_name = smap.get(int(subj))
            if subj_name:
                return f"{subj_name} - Bài {lesson_label}"
            return f"{title} - Bài {lesson_label}" if title else f"{subj} - Bài {lesson_label}"
        return f"{subj} - Bài {lesson_label}"
    return None

# =========================
# Resolve source via BM25 suggestion
# =========================
def _resolve_source_from_bm25_suggestion_with_cache(
    sugg: Dict[str, Any],
    bm_records: List[Dict[str, Any]],
    lesson_cache: Dict[str, Optional[List[Dict[str, Any]]]],
) -> Optional[str]:
    if not sugg or not isinstance(sugg, dict):
        return None

    rec = None
    rid = sugg.get("rid")
    if isinstance(rid, int) and 0 <= rid < len(bm_records):
        rec = bm_records[rid]

    if rec is None and bm_records:
        ans_text = (sugg.get("ans") or sugg.get("answer") or sugg.get("text") or "").strip()
        if ans_text:
            n_ans = _norm_text(ans_text)
            for candidate in bm_records:
                cand_text = ""
                if isinstance(candidate, dict):
                    if isinstance(candidate.get("qa"), dict):
                        cand_text = (candidate.get("qa") or {}).get("answer") or ""
                    cand_text = cand_text or candidate.get("text") or ""
                if cand_text and n_ans and _norm_text(cand_text).find(n_ans) != -1:
                    rec = candidate
                    break

    if rec is None:
        return None

    lesson_id = rec.get("lesson") or rec.get("lesson_id")
    lesson_file = rec.get("file")
    if not lesson_file or lesson_id is None:
        return None

    if lesson_file not in lesson_cache:
        repo_root, lesson_dir, _ = _repo_paths()
        p = lesson_dir / lesson_file
        lesson_cache[lesson_file] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    lf = lesson_cache.get(lesson_file)
    if not isinstance(lf, list):
        return None

    for item in lf:
        lv = (item or {}).get("lesson") or (item or {}).get("lesson_id")
        same = (lv == lesson_id) or (
            isinstance(lv, int) and isinstance(lesson_id, int) and int(lv) == int(lesson_id)
        )
        if not same:
            continue

        subj = (item or {}).get("subject") or (item or {}).get("subject_id") or (item or {}).get("subjectId")
        title = (item or {}).get("tittle") or (item or {}).get("title")
        if subj is None:
            return f"{title} - Bài {lesson_id}" if title else None

        if isinstance(subj, int) or (isinstance(subj, str) and str(subj).isdigit()):
            smap = _load_subject_map()
            subj_name = smap.get(int(subj))
            if subj_name:
                return f"{subj_name} - Bài {lesson_id}"
            return f"{title} - Bài {lesson_id}" if title else f"{subj} - Bài {lesson_id}"
        return f"{subj} - Bài {lesson_id}"
    return None

# =========================
# Small struct builder
# =========================
def _ans_item(ans: Any, src: Optional[str]) -> Dict[str, Any]:
    return {"answer": str(ans) if ans is not None else "", "source": str(src) if src else ""}

# =========================
# Core normalizer
# =========================
def _build_id_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in records or []:
        rid_val: Optional[int] = None
        if isinstance(r.get("rid"), int):
            rid_val = r.get("rid")
        elif isinstance(r.get("rid"), str) and r.get("rid").isdigit():
            rid_val = int(r.get("rid"))
        elif isinstance(r.get("id"), int):
            rid_val = r.get("id")
        elif isinstance(r.get("id"), str) and r.get("id").isdigit():
            rid_val = int(r.get("id"))

        if rid_val is not None:
            idx[rid_val] = r
            idx[str(rid_val)] = r
        else:
            sid = r.get("id") or r.get("rid")
            if isinstance(sid, str) and sid:
                idx[sid] = r
    return idx

def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(input_data, dict):
        return {"status": Status.FAIL.name, "metadata": [], "error": "invalid_input"}

    p_status = (input_data.get("status") or "").strip()
    bm_result = input_data.get("bm_result")
    suggestions = input_data.get("suggestions") or input_data.get("bm25_meta") or []
    sem_result = input_data.get("sem_result") or input_data.get("sem_meta") or []

    # FAST
    if p_status == "definite_exam_answer":
        compare_input = {"thres": 0.0, "metadata": suggestions if isinstance(suggestions, list) else []}
        cmp_res = compare_score_multi_exam_service(compare_input)
        if cmp_res and cmp_res.get("command_bm25") == "BM25_FAST_PATH_DEFINITE_ANSWER":
            if bm_result:
                return {"status": Status.FAST.name, "metadata": [{"answer": str(bm_result)}]}
            if suggestions:
                first = suggestions[0]
                txt = first.get("ans") or first.get("answer") or first.get("text") or ""
                return {"status": Status.FAST.name, "metadata": [{"answer": str(txt)}]}
            return {"status": Status.FAST.name, "metadata": []}

        if bm_result:
            return {"status": Status.FAST.name, "metadata": [{"answer": str(bm_result)}]}
        if suggestions:
            first = suggestions[0]
            txt = first.get("ans") or first.get("answer") or first.get("text") or ""
            return {"status": Status.FAST.name, "metadata": [{"answer": str(txt)}]}
        return {"status": Status.FAST.name, "metadata": []}

    # DEFINITE
    if p_status == "definite_retrieve_answer":
        answer_txt = ""
        src = None
        cand = sem_result[0] if isinstance(sem_result, list) and len(sem_result) > 0 else None

        if cand:
            answer_txt = (cand.get("text") or cand.get("ans") or cand.get("answer") or "").strip()
            meta = cand.get("meta") if isinstance(cand, dict) else None
            src = _build_source_from_meta(meta) or _resolve_source_from_lesson_meta(meta)

        if not src and cand:
            _, sem_records = load_data_semantic_mock.load_corpus()
            id_index = _build_id_index(sem_records)
            mapped = _map_suggestion_to_semantic_source({"ans": answer_txt, "rid": cand.get("rid")}, sem_records, id_index)
            if mapped:
                src = mapped

        if not src:
            _, sem_records2 = load_data_semantic_mock.load_corpus()
            id_index2 = _build_id_index(sem_records2)
            try:
                bm_records2, _ = __import__("services.load_data_BM25_mock", fromlist=["load_data_bm25"]).load_data_bm25()
            except Exception:
                # Trường hợp module mock không tồn tại ở môi trường prod — không dùng try/except ở logic,
                # nhưng vẫn giữ fallback đơn giản để không phá API khi thiếu file mock.
                bm_records2 = []
            lesson_cache2: Dict[str, Optional[List[Dict[str, Any]]]] = {}
            for s in (suggestions or [])[:3]:
                src_try = _resolve_source_from_bm25_suggestion_with_cache(s, bm_records2, lesson_cache2)
                if not src_try:
                    src_try = _map_suggestion_to_semantic_source(s, sem_records2 or [], id_index2)
                if src_try:
                    src = src_try
                    break

        if not src and answer_txt:
            try:
                bm_records, _ = __import__("services.load_data_BM25_mock", fromlist=["load_data_bm25"]).load_data_bm25()
            except Exception:
                bm_records = []
            n_ans = _norm_text(answer_txt)
            for rec in bm_records:
                cand_text = ""
                if isinstance(rec, dict):
                    if isinstance(rec.get("qa"), dict):
                        cand_text = (rec.get("qa") or {}).get("answer") or ""
                    cand_text = cand_text or rec.get("text") or ""
                if cand_text and n_ans and _norm_text(cand_text).find(n_ans) != -1:
                    meta = {k: rec.get(k) for k in ("subject", "subject_id", "lesson", "lesson_id", "file") if k in rec}
                    src_try = _build_source_from_meta(meta)
                    if not src_try:
                        src_try = _resolve_source_from_bm25_suggestion_with_cache({"rid": rec.get("rid")}, bm_records, {})
                    if src_try:
                        src = src_try
                        break

        if not answer_txt:
            if bm_result:
                answer_txt = str(bm_result)
            elif suggestions:
                answer_txt = (suggestions[0].get("ans") or suggestions[0].get("answer") or suggestions[0].get("text") or "").strip()
            else:
                answer_txt = ""

        if not src:
            logging.getLogger(__name__).debug("no source resolved for definite candidate")
        return {"status": Status.DEFINITE.name, "metadata": [_ans_item(answer_txt, src)]}

    # POSSIBLE
    if p_status == "suggest_top3_bm25":
        _, sem_records = load_data_semantic_mock.load_corpus()
        id_index = _build_id_index(sem_records)
        try:
            bm_records, _ = __import__("services.load_data_BM25_mock", fromlist=["load_data_bm25"]).load_data_bm25()
        except Exception:
            bm_records = []

        lesson_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        out: List[Dict[str, str]] = []
        for s in (suggestions or [])[:3]:
            ans_text = (s.get("ans") or s.get("answer") or s.get("text") or "").strip()
            src = _resolve_source_from_bm25_suggestion_with_cache(s, bm_records, lesson_cache)
            if not src:
                src = _map_suggestion_to_semantic_source(s, sem_records, id_index)
            out.append({"answer": ans_text, "source": src or ""})
        return {"status": Status.POSSIBLE.name, "metadata": out}

    # UNKNOW
    if p_status == "no_bm25_result":
        return {"status": Status.UNKNOW.name, "metadata": [{"answer": "KHÔNG CÓ CÂU TRẢ LỜI.", "source":""}]}

    # fallback
    return {"status": Status.FAIL.name, "metadata": "ĐÃ XẢY RA LỖI."}

# =========================
# Convenience wrapper
# =========================
def run_text(text: str, top_k: int = 5) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"status": Status.FAIL.name, "text": "", "result": {"metadata": []}, "error": "empty_input"}

    pipeline_input = {"text": text, "top_k": int(top_k)}
    # gọi trực tiếp (không try/except): nếu pipeline lỗi sẽ văng sớm
    pipe_out = get_results_bm25_and_semantic_service.run(pipeline_input)

    return run(pipe_out)
