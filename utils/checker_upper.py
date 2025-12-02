from PanoHandler import match_pii, PanoPlus
from datasets import load_from_disk  # or from datasets import load_dataset

dataset = load_from_disk("PANOPTICON_v1.2")

print(dataset[8774]['PromptPII'])
