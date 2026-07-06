# reasoning/analyzer.py
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
import re

import pandas as pd


@dataclass
class PatternAnalysis:
    invariant_candidates: List[str]
    varying_candidates: List[str]
    query_entity: str
    transfer_hypothesis: str
    confidence: float

    # backward compatibility
    common_elements: List[str] = field(default_factory=list)
    support_values: List[str] = field(default_factory=list)
    query_values: List[str] = field(default_factory=list)
    relation_type: str = "unknown"
    notes: str = ""


class ICLPatternAnalyzer:
    """
    Benchmark-agnostic analyzer.

    It does NOT use:
    - task_id
    - x_space / theta_space
    - CoBSAT task names as labels for logic

    It infers the pattern from:
    - short text labels
    - image file names
    - optional captions / descriptions / CSV caption lookup

    Core principle:
    use structural agreement across demonstrations, not benchmark metadata.
    """

    STOPWORDS = {
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "to", "of", "for", "with", "by", "from",
        "as", "and", "or", "but", "if", "then", "than", "it", "its",
        "their", "there", "here", "which", "who", "whom", "whose",
        "image", "images", "scene", "shown", "depicted", "feature",
        "features", "appears", "appearing", "visible", "overall",
        "main", "focus", "some", "others",
    }

    LIGHT_WORDS = {
        "large", "small", "big", "little", "simple", "single",
        "various", "multiple", "overall", "main", "middle",
        "front", "back", "top", "bottom",
    }

    GENERIC_HEADS = {
        "people", "person", "group", "thing", "object", "objects",
        "image", "scene", "background", "foreground", "atmosphere",
        "surface", "area", "place",
    }

    def __init__(
        self,
        datasets_root: Optional[str] = None,
        allow_caption_lookup: bool = True,
    ):
        self.datasets_root = Path(datasets_root) if datasets_root else None
        self.allow_caption_lookup = allow_caption_lookup
        self._table_cache: Dict[Path, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # basic normalization
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        text = str(text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _normalize_label(self, text: str) -> str:
        text = self._normalize(text)
        text = text.rstrip(":").strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9\-']+", self._normalize(text))

    def _content_words(self, text: str) -> List[str]:
        return [
            t for t in self._tokenize(text)
            if t not in self.STOPWORDS
            and t not in self.LIGHT_WORDS
            and len(t) > 2
        ]

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"[.!?\n]+", self._normalize(text))
        return [p.strip() for p in parts if p.strip()]

    def _is_short_label(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False

        toks = self._tokenize(raw)
        if not toks:
            return False

        if raw.endswith(":") and len(toks) <= 3:
            return True
        if len(toks) == 1:
            return True
        return False

    # ------------------------------------------------------------------
    # sample parsing
    # ------------------------------------------------------------------

    def _build_examples_from_inputs(self, sample_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        text_inputs = sample_dict.get("text_inputs", [])
        image_inputs = sample_dict.get("image_inputs", [])

        if isinstance(text_inputs, list) and text_inputs:
            demo_count = max(0, len(text_inputs) - 1)
        elif isinstance(image_inputs, list):
            demo_count = len(image_inputs)
        else:
            demo_count = 0

        examples: List[Dict[str, Any]] = []
        for i in range(demo_count):
            ex = {"idx": i}
            if isinstance(text_inputs, list) and i < len(text_inputs) - 1:
                ex["text"] = text_inputs[i]
            if isinstance(image_inputs, list) and i < len(image_inputs):
                ex["image"] = image_inputs[i]
            examples.append(ex)

        return examples

    def _extract_examples(self, sample_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        examples = sample_dict.get("examples", [])
        if isinstance(examples, list) and examples:
            return [ex for ex in examples if isinstance(ex, dict)]
        return self._build_examples_from_inputs(sample_dict)

    def _extract_support_values(self, sample_dict: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        for ex in self._extract_examples(sample_dict):
            txt = ex.get("text", "")
            if self._is_short_label(txt):
                label = self._normalize_label(txt)
                if label and label not in values:
                    values.append(label)
        return values

    def _extract_query_values(self, sample_dict: Dict[str, Any]) -> List[str]:
        query_values: List[str] = []

        query_block = sample_dict.get("query", {})
        if isinstance(query_block, dict):
            q = query_block.get("text", "")
            if isinstance(q, str) and q.strip():
                query_values.append(self._normalize_label(q))

        if query_values:
            return query_values

        text_inputs = sample_dict.get("text_inputs", [])
        if isinstance(text_inputs, list) and text_inputs:
            q = text_inputs[-1]
            if isinstance(q, str) and q.strip():
                query_values.append(self._normalize_label(q))

        return query_values

    def _extract_query_entity(self, sample_dict: Dict[str, Any]) -> str:
        qv = self._extract_query_values(sample_dict)
        return qv[0] if qv else ""

    # ------------------------------------------------------------------
    # caption loading
    # ------------------------------------------------------------------

    def _load_table(self, table_path: Path) -> pd.DataFrame:
        if table_path in self._table_cache:
            return self._table_cache[table_path]

        if not table_path.exists():
            raise FileNotFoundError(f"Table file not found: {table_path}")

        df = pd.read_csv(table_path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "image" not in df.columns or "caption" not in df.columns:
            raise ValueError(
                f"{table_path} must contain columns 'image' and 'caption'. "
                f"Found: {list(df.columns)}"
            )

        df["image"] = df["image"].astype(str).str.strip()
        df["image_basename"] = df["image"].apply(lambda x: Path(x).name.lower())
        df["caption"] = df["caption"].astype(str).fillna("").str.strip()

        self._table_cache[table_path] = df
        return df

    def _lookup_caption(self, image_path: str) -> str:
        if not self.datasets_root or not image_path:
            return ""

        try:
            p = Path(image_path)
            folder = p.parent.name.lower()

            if "_" not in folder:
                return ""

            prefix = folder.split("_", 1)[0]
            candidate_files = [
                self.datasets_root / f"{prefix}_animal.csv",
                self.datasets_root / f"{prefix}_object.csv",
            ]

            basename = p.name.lower()

            for table_path in candidate_files:
                if not table_path.exists():
                    continue

                try:
                    df = self._load_table(table_path)
                except Exception:
                    continue

                row = df[df["image_basename"] == basename]
                if not row.empty:
                    caption = str(row.iloc[0]["caption"]).strip()
                    if caption and caption.lower() != "nan":
                        return caption
        except Exception:
            pass

        return ""

    def _collect_text_sources(self, ex: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Returns list of (text, source_type)
        source_type:
        - explicit_caption
        - explicit_text
        - lookup_caption
        - short_label
        """
        sources: List[Tuple[str, str]] = []

        raw_text = ex.get("text", "")
        if isinstance(raw_text, str) and raw_text.strip():
            if self._is_short_label(raw_text):
                label = self._normalize_label(raw_text)
                if label:
                    sources.append((label, "short_label"))
            else:
                sources.append((raw_text.strip(), "explicit_text"))

        for key in ["caption", "image_caption", "description", "desc", "text_description"]:
            val = ex.get(key)
            if isinstance(val, str) and val.strip():
                sources.append((val.strip(), "explicit_caption"))
                break

        if self.allow_caption_lookup:
            img = ex.get("image")
            if isinstance(img, str) and img.strip():
                cap = self._lookup_caption(img)
                if cap:
                    sources.append((cap, "lookup_caption"))

        seen = set()
        deduped = []
        for txt, src in sources:
            key = (self._normalize(txt), src)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((txt, src))

        return deduped

    # ------------------------------------------------------------------
    # image-name structural analysis
    # ------------------------------------------------------------------

    def _image_stem_tokens(self, image_path: str) -> List[str]:
        if not image_path:
            return []
        stem = Path(str(image_path)).stem.lower()
        parts = re.split(r"[_\-\s]+", stem)
        return [p for p in parts if p]

    def _infer_from_image_names(
        self,
        examples: List[Dict[str, Any]],
        support_values: List[str],
        query_entity: str,
    ) -> Dict[str, Any]:
        """
        Core dataset-agnostic structural inference from example file names.

        Typical cases:
        - blue_car, brown_car, query orange -> shared suffix car => varying attribute
        - park_cat, park_cow, query lion -> shared prefix park => varying subject
        """
        stem_lists = []
        for ex in examples:
            img = ex.get("image", "")
            toks = self._image_stem_tokens(img)
            if toks:
                stem_lists.append(toks)

        result = {
            "mode": None,                 # "attribute_value" | "subject_identity" | None
            "invariants": [],
            "evidence": [],
        }

        if len(stem_lists) < 2:
            return result

        # only compare when token lengths are aligned
        min_len = min(len(x) for x in stem_lists)
        if min_len == 0:
            return result

        same_by_position = []
        for pos in range(min_len):
            vals = [tokens[pos] for tokens in stem_lists]
            if len(set(vals)) == 1:
                same_by_position.append((pos, vals[0]))

        support_set = {self._normalize_label(x) for x in support_values if x}
        query_norm = self._normalize_label(query_entity)

        # Case A: shared suffix -> usually stable entity/object, labels vary as attribute values
        suffix_token = None
        last_vals = [tokens[-1] for tokens in stem_lists]
        if len(set(last_vals)) == 1:
            suffix_token = last_vals[0]

        if suffix_token and suffix_token not in support_set and suffix_token != query_norm:
            result["mode"] = "attribute_value"
            result["invariants"] = [suffix_token]
            result["evidence"].append(f"shared image-name suffix: {suffix_token}")
            return result

        # Case B: shared prefix -> usually stable scene/context, labels vary as entity
        prefix_token = None
        first_vals = [tokens[0] for tokens in stem_lists]
        if len(set(first_vals)) == 1:
            prefix_token = first_vals[0]

        if prefix_token and prefix_token not in support_set and prefix_token != query_norm:
            result["mode"] = "subject_identity"
            result["invariants"] = [prefix_token]
            result["evidence"].append(f"shared image-name prefix: {prefix_token}")
            return result

        # Case C: any shared positional token not equal to labels/query
        shared_tokens = [tok for _, tok in same_by_position if tok not in support_set and tok != query_norm]
        if shared_tokens:
            # if shared token is near the end, treat as stable entity/object
            best_pos, best_tok = same_by_position[-1]
            if best_tok not in support_set and best_tok != query_norm:
                if best_pos >= min_len - 2:
                    result["mode"] = "attribute_value"
                else:
                    result["mode"] = "subject_identity"
                result["invariants"] = [best_tok]
                result["evidence"].append(f"shared image-name token at position {best_pos}: {best_tok}")
                return result

        return result

    # ------------------------------------------------------------------
    # caption fallback analysis
    # ------------------------------------------------------------------

    def _extract_candidate_phrases(self, text: str) -> List[Tuple[str, int]]:
        text = self._normalize(text)
        if not text:
            return []

        candidates: List[Tuple[str, int]] = []
        seen = set()

        sentences = self._split_sentences(text)
        for sent_idx, sent in enumerate(sentences):
            words = self._content_words(sent)
            if not words:
                continue

            for n in (1, 2, 3):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n]).strip()
                    if not phrase or phrase in seen:
                        continue
                    seen.add(phrase)
                    candidates.append((phrase, sent_idx))

            rel_patterns = [
                r"(in front of [a-z0-9\s\-']{3,40})",
                r"(with [a-z0-9\s\-']{3,40} in the background)",
                r"(surrounded by [a-z0-9\s\-']{3,40})",
            ]
            for pat in rel_patterns:
                for m in re.finditer(pat, sent):
                    phrase = self._normalize(m.group(1))
                    phrase = re.sub(r"\s+", " ", phrase).strip()
                    if not phrase or phrase in seen:
                        continue
                    seen.add(phrase)
                    candidates.append((phrase, sent_idx))

        return candidates

    def _extract_demo_features(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        demo_features: List[Dict[str, Any]] = []

        for ex in examples:
            short_label = self._normalize_label(ex.get("text", "")) if self._is_short_label(ex.get("text", "")) else ""
            text_sources = self._collect_text_sources(ex)

            phrase_entries = []
            for txt, src in text_sources:
                if src == "short_label":
                    continue
                for phrase, sent_idx in self._extract_candidate_phrases(txt):
                    phrase_entries.append(
                        {
                            "phrase": self._normalize(phrase),
                            "sent_idx": sent_idx,
                            "source_type": src,
                        }
                    )

            demo_features.append(
                {
                    "short_label": short_label,
                    "text_sources": text_sources,
                    "phrase_entries": phrase_entries,
                }
            )

        return demo_features

    def _score_phrase(
        self,
        phrase: str,
        occurrences: List[Dict[str, Any]],
        total_demos: int,
        support_values: List[str],
    ) -> float:
        toks = self._tokenize(phrase)
        if not toks:
            return 0.0

        support_set = {self._normalize_label(x) for x in support_values if x}
        support_heads = {self._tokenize(x)[-1] for x in support_set if self._tokenize(x)}

        coverage = len({o["demo_idx"] for o in occurrences}) / max(1, total_demos)
        mean_sent_idx = sum(o["sent_idx"] for o in occurrences) / max(1, len(occurrences))

        source_counts = Counter(o["source_type"] for o in occurrences)
        explicit_sources = source_counts.get("explicit_caption", 0) + source_counts.get("explicit_text", 0)
        lookup_sources = source_counts.get("lookup_caption", 0)

        score = 0.0
        score += 0.44 * coverage

        if mean_sent_idx <= 0.5:
            score += 0.08
        elif mean_sent_idx <= 1.5:
            score += 0.04

        if len(toks) == 1:
            score -= 0.10
        elif len(toks) in (2, 3):
            score += 0.10
        elif len(toks) >= 5:
            score -= 0.05

        if explicit_sources > 0:
            score += 0.16
        if lookup_sources > 0:
            score += 0.04
        if lookup_sources > 0 and explicit_sources == 0:
            score -= 0.18

        norm_phrase = self._normalize(phrase)
        head = toks[-1]

        if norm_phrase in support_set:
            score -= 0.55
        if head in support_heads:
            score -= 0.28

        if len(toks) == 1 and head in self.GENERIC_HEADS:
            score -= 0.20

        return max(0.0, min(1.0, score))

    def _rank_shared_caption_candidates(
        self,
        demo_features: List[Dict[str, Any]],
        support_values: List[str],
    ) -> List[Tuple[str, float]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for demo_idx, feat in enumerate(demo_features):
            seen_local = set()
            for entry in feat["phrase_entries"]:
                phrase = entry["phrase"]
                if not phrase or phrase in seen_local:
                    continue
                seen_local.add(phrase)
                grouped[phrase].append(
                    {
                        "demo_idx": demo_idx,
                        "sent_idx": entry["sent_idx"],
                        "source_type": entry["source_type"],
                    }
                )

        total_demos = len(demo_features)
        ranked: List[Tuple[str, float]] = []

        for phrase, occs in grouped.items():
            score = self._score_phrase(
                phrase=phrase,
                occurrences=occs,
                total_demos=total_demos,
                support_values=support_values,
            )
            ranked.append((phrase, score))

        ranked.sort(key=lambda x: (x[1], len(x[0].split()), len(x[0])), reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # high-level inference
    # ------------------------------------------------------------------

    def _infer_structure(
        self,
        examples: List[Dict[str, Any]],
        support_values: List[str],
        query_entity: str,
    ) -> Dict[str, Any]:
        """
        Main logic:
        1) try image-name structural inference
        2) fallback to captions
        """
        image_name_result = self._infer_from_image_names(examples, support_values, query_entity)
        demo_features = self._extract_demo_features(examples)
        ranked_caption = self._rank_shared_caption_candidates(demo_features, support_values)

        invariants: List[str] = []
        weak_shared: List[str] = []
        varying: List[str] = []
        evidence: List[str] = []

        if image_name_result["mode"] == "attribute_value":
            invariants = image_name_result["invariants"][:]
            varying = ["attribute_value"]
            evidence.extend(image_name_result["evidence"])
        elif image_name_result["mode"] == "subject_identity":
            invariants = image_name_result["invariants"][:]
            varying = ["subject_identity"]
            evidence.extend(image_name_result["evidence"])
        else:
            # caption-only fallback
            weak_shared = [p for p, s in ranked_caption if 0.40 <= s < 0.72][:6]
            invariants = [p for p, s in ranked_caption if s >= 0.72][:3]

        # if image-name inference already decided mode, captions only add weak notes
        if image_name_result["mode"] is not None:
            weak_shared = [p for p, s in ranked_caption if 0.40 <= s < 0.72][:6]
            if not invariants and weak_shared:
                invariants = ["shared_context"]

        # detect subject variation from labels if not already present
        labels = [self._normalize_label(x) for x in support_values if x]
        if len(set(labels)) > 1:
            if "attribute_value" not in varying and "subject_identity" not in varying:
                # choose safer default if no structural signal
                varying.append("subject_identity")

        # add generic detail variation if examples contain many non-shared caption fragments
        if ranked_caption:
            phrase_sets = []
            invariant_set = {self._normalize(x) for x in invariants}
            for feat in demo_features:
                phrase_set = {
                    entry["phrase"]
                    for entry in feat["phrase_entries"]
                    if entry["phrase"] not in invariant_set
                }
                phrase_sets.append(phrase_set)

            if len(phrase_sets) >= 2:
                union_phrases = set.union(*phrase_sets) if phrase_sets else set()
                if len(union_phrases) >= 3:
                    varying.append("demo_specific_details")

        # If we only have weak caption overlap and no structural inference,
        # expose a safe invariant rather than a noisy raw phrase.
        if not invariants and weak_shared:
            invariants = ["shared_context"]

        return {
            "invariants": invariants[:4],
            "weak_shared": weak_shared[:6],
            "varying": sorted(set(varying)),
            "evidence": evidence,
            "ranked_caption": ranked_caption,
            "demo_features": demo_features,
        }

    def _infer_transfer_hypothesis(
        self,
        invariants: List[str],
        weak_shared: List[str],
        varyings: List[str],
        query_entity: str,
    ) -> str:
        if "attribute_value" in varyings:
            if invariants:
                return (
                    f"Preserve the shared elements ({', '.join(invariants[:3])}) "
                    f"and apply the query value ({query_entity}) as the varying attribute."
                )
            return (
                f"Use the demonstrations to infer the varying attribute and apply "
                f"the query value ({query_entity}) while preserving other supported structure."
            )

        if "subject_identity" in varyings and invariants:
            return (
                f"Preserve the shared elements ({', '.join(invariants[:3])}) "
                f"and replace the subject with the query entity ({query_entity})."
            )

        if "subject_identity" in varyings and weak_shared:
            return (
                f"Replace the demonstrated subject with the query entity ({query_entity}). "
                f"Repeated weak shared cues ({', '.join(weak_shared[:3])}) may describe context, "
                f"but should not be treated as mandatory."
            )

        if invariants:
            return (
                f"Preserve the shared elements ({', '.join(invariants[:3])}) "
                f"while adapting to the query entity ({query_entity})."
            )

        if varyings:
            return (
                f"Transfer the varying aspects ({', '.join(varyings)}) "
                f"to the query entity ({query_entity}) without introducing unrelated details."
            )

        return f"Adapt the demonstrated pattern to the query entity ({query_entity})."

    def _estimate_confidence(
        self,
        support_values: List[str],
        invariants: List[str],
        weak_shared: List[str],
        varyings: List[str],
        evidence: List[str],
        demo_features: List[Dict[str, Any]],
    ) -> float:
        score = 0.35

        if evidence:
            score += 0.20

        if invariants:
            score += 0.14
        elif weak_shared:
            score += 0.05

        if varyings:
            score += 0.12

        if "attribute_value" in varyings or "subject_identity" in varyings:
            score += 0.08

        lookup_count = 0
        explicit_count = 0
        for feat in demo_features:
            srcs = {src for _, src in feat["text_sources"]}
            if "lookup_caption" in srcs:
                lookup_count += 1
            if "explicit_caption" in srcs or "explicit_text" in srcs:
                explicit_count += 1

        if explicit_count >= 2:
            score += 0.08
        elif lookup_count >= 2:
            score += 0.04

        return max(0.0, min(0.90, score))

    def _infer_relation_type(
        self,
        invariants: List[str],
        varyings: List[str],
        query_entity: str,
    ) -> str:
        if invariants and varyings and query_entity:
            return "structured-transfer"
        if invariants and query_entity:
            return "invariant-preserving-transfer"
        if varyings and query_entity:
            return "variation-transfer"
        if query_entity:
            return "set-to-query-transfer"
        return "unknown"

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def analyze_icl_patterns(self, sample_dict: Dict[str, Any]) -> PatternAnalysis:
        examples = self._extract_examples(sample_dict)
        support_values = self._extract_support_values(sample_dict)
        query_values = self._extract_query_values(sample_dict)
        query_entity = self._extract_query_entity(sample_dict)

        structure = self._infer_structure(
            examples=examples,
            support_values=support_values,
            query_entity=query_entity,
        )

        invariants = structure["invariants"]
        weak_shared = structure["weak_shared"]
        varyings = structure["varying"]
        evidence = structure["evidence"]
        ranked_caption = structure["ranked_caption"]
        demo_features = structure["demo_features"]

        transfer_hypothesis = self._infer_transfer_hypothesis(
            invariants=invariants,
            weak_shared=weak_shared,
            varyings=varyings,
            query_entity=query_entity,
        )

        confidence = self._estimate_confidence(
            support_values=support_values,
            invariants=invariants,
            weak_shared=weak_shared,
            varyings=varyings,
            evidence=evidence,
            demo_features=demo_features,
        )

        relation_type = self._infer_relation_type(
            invariants=invariants,
            varyings=varyings,
            query_entity=query_entity,
        )

        lookup_count = 0
        explicit_count = 0
        for feat in demo_features:
            srcs = {src for _, src in feat["text_sources"]}
            if "lookup_caption" in srcs:
                lookup_count += 1
            if "explicit_caption" in srcs or "explicit_text" in srcs:
                explicit_count += 1

        notes_parts = [
            f"Extracted from {len(examples)} demonstrations.",
            f"Explicit text evidence: {explicit_count}/{len(examples)}.",
            f"Lookup captions used: {lookup_count}/{len(examples)}.",
            "No task metadata was used.",
        ]

        if evidence:
            notes_parts.append(f"Structural evidence: {evidence}.")

        if invariants:
            notes_parts.append(f"Invariant candidates: {invariants}.")
        else:
            notes_parts.append("No high-confidence invariant candidates were inferred.")

        if weak_shared:
            notes_parts.append(
                f"Weak shared cues (not promoted directly to invariants): {weak_shared[:5]}."
            )

        if varyings:
            notes_parts.append(f"Varying candidates: {varyings}.")
        else:
            notes_parts.append("No strong varying candidates were inferred.")

        top_scores = [f"{p}:{round(s, 3)}" for p, s in ranked_caption[:5]]
        if top_scores:
            notes_parts.append(f"Top caption cue scores: {top_scores}.")

        notes = " ".join(notes_parts)

        return PatternAnalysis(
            invariant_candidates=invariants,
            varying_candidates=varyings,
            query_entity=query_entity,
            transfer_hypothesis=transfer_hypothesis,
            confidence=confidence,
            common_elements=invariants[:],
            support_values=support_values,
            query_values=query_values,
            relation_type=relation_type,
            notes=notes,
        )