from PanoHandler import match_pii, PanoPlus
from datasets import load_from_disk  # or from datasets import load_dataset

pano_plus = PanoPlus()
pano_plus_by_id = {row["Unique ID"]: row for row in pano_plus.dataset}

dataset = load_from_disk("PANOPTICON")  # or load_dataset(..., split="train")

def fix_pii(example):
    prompt = example["prompt"]
    uid = example["ID"]

    full_pii = pano_plus_by_id[uid]
    matched = match_pii(prompt, full_pii)

    example["PromptPII"] = matched  # or json.dumps(matched)
    return example

dataset_fixed = dataset.map(
    fix_pii,
    load_from_cache_file=False,
    desc="Recomputing PromptPII with fixed matcher",
)


print('fixed!')