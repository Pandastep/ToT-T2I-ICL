import json
import re
from pathlib import Path

ROOT = Path("results_ablation")

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
    pattern = rf"{re.escape(field)}:\s*(.*?)(?:;|$)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""

summary_paths = sorted(ROOT.glob("task*_branch_B*/task_*/summary.json"))

created = 0
skipped = 0

for summary_path in summary_paths:
    task_dir = summary_path.parent
    summary = read_json(summary_path)
    samples = summary.get("samples", [])

    parsed = []
    object_pool = []

    for sample in samples:
        sample_idx = int(sample["sample_idx"])
        hs = sample.get("hypothesis_summary", "")

        target_object = clean(extract_field(hs, "Invariants"))
        target_attribute = clean(extract_field(hs, "QueryEntity"))
        support_values = split_values(extract_field(hs, "SupportValues"))

        if target_object:
            object_pool.append(target_object)

        parsed.append({
            "sample_idx": sample_idx,
            "target_object": target_object,
            "target_attribute": target_attribute,
            "support_values": support_values,
        })

    object_pool = list(dict.fromkeys([x for x in object_pool if x]))

    for item in parsed:
        sample_idx = item["sample_idx"]
        sample_dir = task_dir / f"sample_{sample_idx:03d}"

        if not sample_dir.exists():
            print(f"Missing sample dir: {sample_dir}")
            skipped += 1
            continue

        target_object = item["target_object"]
        target_attribute = item["target_attribute"]
        support_values = item["support_values"]

        if not target_object or not target_attribute:
            print(f"Missing target metadata: {sample_dir}")
            skipped += 1
            continue

        attribute_pool = list(dict.fromkeys(
            [x for x in support_values + [target_attribute] if x]
        ))

        final_object_pool = list(object_pool)
        if len(final_object_pool) < 2:
            final_object_pool.append("object")

        if len(attribute_pool) < 2:
            attribute_pool.append("unknown")

        meta = {
            "target_object": target_object,
            "target_attribute": target_attribute,
            "object_pool": final_object_pool,
            "attribute_pool": attribute_pool,
        }

        write_json(sample_dir / "csr_meta.json", meta)
        created += 1

print(f"Created csr_meta.json files: {created}")
print(f"Skipped samples: {skipped}")