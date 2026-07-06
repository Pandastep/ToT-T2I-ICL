import json
import re
from pathlib import Path

ROOT = Path("results_ablation") / "weight_sensitivity"

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def clean(x):
    return str(x).strip().lower()

def split_values(x):
    if not x:
        return []
    return [clean(v) for v in re.split(r"[,;|]+", x) if clean(v)]

def extract_field(text, field):
    # Example: Invariants: car; Varyings: attribute_value; ...
    pattern = rf"{re.escape(field)}:\s*(.*?)(?:;|$)"
    m = re.search(pattern, text)
    if not m:
        return ""
    return m.group(1).strip()

summary_paths = sorted(ROOT.glob("task*_weight_*/task_*/summary.json"))

if not summary_paths:
    raise FileNotFoundError(f"No summary.json files found under {ROOT}")

created = 0
skipped = 0

for summary_path in summary_paths:
    summary = read_json(summary_path)
    samples = summary.get("samples", [])

    # Build object pool from all invariants in this task/mode.
    object_pool = []

    parsed_samples = []

    for sample in samples:
        hs = sample.get("hypothesis_summary", "")

        target_object = clean(extract_field(hs, "Invariants"))
        target_attribute = clean(extract_field(hs, "QueryEntity"))
        support_values = split_values(extract_field(hs, "SupportValues"))

        if target_object:
            object_pool.append(target_object)

        parsed_samples.append({
            "sample": sample,
            "target_object": target_object,
            "target_attribute": target_attribute,
            "support_values": support_values,
        })

    # deduplicate object pool
    object_pool = list(dict.fromkeys([x for x in object_pool if x]))

    for item in parsed_samples:
        sample = item["sample"]
        target_object = item["target_object"]
        target_attribute = item["target_attribute"]
        support_values = item["support_values"]

        if not target_object or not target_attribute:
            skipped += 1
            continue

        attribute_pool = list(dict.fromkeys(
            [x for x in support_values + [target_attribute] if x]
        ))

        # CSR evaluator requires at least 2 candidates.
        # If this task has only one object in object_pool, add a weak dummy distractor.
        # Usually with 6 samples there should be several objects.
        final_object_pool = list(object_pool)
        if len(final_object_pool) < 2:
            final_object_pool.append("object")

        if len(attribute_pool) < 2:
            attribute_pool.append("unknown")

        sample_dir = Path(sample["prompt_file"]).parent
        out_path = sample_dir / "csr_meta.json"

        meta = {
            "target_object": target_object,
            "target_attribute": target_attribute,
            "object_pool": final_object_pool,
            "attribute_pool": attribute_pool,
        }

        write_json(out_path, meta)
        created += 1

print(f"Created csr_meta.json files: {created}")
print(f"Skipped samples: {skipped}")