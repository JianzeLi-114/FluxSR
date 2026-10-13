import json
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Reads prompts from a JSON file and yields each as a photo-quality-
    augmented string suitable for FLUX text conditioning.

    The JSON may be either a list of raw strings or a list of dicts with a
    ``"prompt"`` key. The default file is ``weights/fluxsr/prompts_10k.json``;
    user-provided files in either format are also supported.
    """

    QUALITY_SUFFIX = (
        ', best quality, masterpiece, ultra high res, '
        '(photorealistic:1.4), RAW photo'
    )

    def __init__(self, json_file="weights/fluxsr/prompts_10k.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"{json_file}: expected non-empty JSON list")

        first = raw[0]
        if isinstance(first, str):
            self.data = raw
        elif isinstance(first, dict) and 'prompt' in first:
            self.data = [item['prompt'] for item in raw]
        else:
            raise ValueError(
                f"{json_file}: entries must be str or {{'prompt': str}}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range (0, {len(self.data)})")

        # FLUX's T5 encoder handles long prompts (up to 512 tokens); only clip
        # as a loose safety cap, then append fixed quality tags.
        words = self.data[idx].split()[:220]
        return ' '.join(words) + self.QUALITY_SUFFIX
