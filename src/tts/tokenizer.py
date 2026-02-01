import json
from pathlib import Path
from typing import List

from phonemizer.backend import EspeakBackend

import src.tts.config as config


class Tokenizer:
    def __init__(self, vocab_path=config.VOCAB_PATH):
        self.vocab_path: Path = vocab_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.eos = "<EOS>"
        self.pad = "<PAD>"
        self.special_tokens = [self.eos, self.pad]

        self.backend = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        if self.vocab_path.exists():
            with open(self.vocab_path, "r") as f:
                self.token_to_id = json.load(f)
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str) -> List[int]:
        phs = self.backend.phonemize(
            [text],
            strip=True,
        )[0]

        ids = [self.token_to_id[ph] for ph in phs]
        ids.append(self.token_to_id[self.eos])

        return ids

    def build_from_text(self, texts: List[str]):
        all_ph = self.backend.phonemize(
            texts,
            strip=True,
        )

        ph_unique = set()
        for ph_str in all_ph:
            ph_unique.update(list(ph_str))

        ph_unique = sorted(list(ph_unique))
        cur_id = config.PHONEME_START_ID
        self.token_to_id = {}

        for tok in self.special_tokens:
            self.token_to_id[tok] = cur_id
            cur_id += 1

        for tok in ph_unique:
            self.token_to_id[tok] = cur_id
            cur_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        with open(self.vocab_path, "w") as f:
            json.dump(self.token_to_id, f, indent=4, ensure_ascii=False)
