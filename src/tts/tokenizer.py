import json
from pathlib import Path
from typing import List

from phonemizer import phonemize

import src.tts.config as config


class Tokenizer:
    def __init__(self, vocab_path=config.VOCAB_PATH):
        self.vocab_path: Path = vocab_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        self.special_tokens = [self.eos, self.pad]

        if self.vocab_path.exists():
            with open(self.vocab_path, "r") as f:
                self.token_to_id = json.load(f)
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str) -> List[int]:
        tokens = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )

        ids = [self.token_to_id[p] for p in tokens]
        ids.append(self.token_to_id[self.eos])

        return ids

    def build_from_text(self, texts: List[str]):
        text = " ".join(texts)
        all_ph = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        ph_unique = sorted(list(set(all_ph)))
        cur_id = config.PHONEME_START_ID

        for tok in self.special_tokens:
            self.token_to_id[tok] = cur_id
            cur_id += 1

        for tok in ph_unique:
            self.token_to_id[tok] = cur_id
            cur_id += 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        with open(self.vocab_path, "w") as f:
            json.dump(self.token_to_id, f, indent=4, ensure_ascii=False)
