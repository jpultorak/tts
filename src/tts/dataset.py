from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import tts.config as config


class TTSDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(list(Path(data_dir).glob("*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path)

        phonemes = data["phonemes"].long()
        audio = data["audio_tokens"].long()
        
        sep = torch.tensor([config.SEP_TOKEN_ID], dtype=torch.long)
        eos = torch.tensor([config.EOS_TOKEN_ID], dtype=torch.long)

        full_seq = torch.cat([phonemes, sep, audio, eos])

        x = full_seq[:-1]
        y = full_seq[1:]

        context_len = len(phonemes) + 1

        labels = y.clone()
        labels[: context_len - 1] = -100

        return x, labels


def collate_fn(batch):
    xs, ys = zip(*batch)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)

    return x_padded, y_padded
