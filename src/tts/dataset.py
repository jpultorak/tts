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

        src = phonemes

        sep = torch.tensor([config.SEP_TOKEN_ID], dtype=torch.long)
        tgt_input = torch.cat([sep, audio])

        eos = torch.tensor([config.EOS_TOKEN_ID], dtype=torch.long)
        tgt_output = torch.cat([audio, eos])

        return src, tgt_input, tgt_output


def collate_fn(batch):
    srcs, tgt_ins, tgt_outs = zip(*batch)

    src_padded = pad_sequence(srcs, batch_first=True, padding_value=config.PAD_TOKEN_ID)
    tgt_in_padded = pad_sequence(
        tgt_ins, batch_first=True, padding_value=config.PAD_TOKEN_ID
    )
    tgt_out_padded = pad_sequence(tgt_outs, batch_first=True, padding_value=-100)

    return src_padded, tgt_in_padded, tgt_out_padded
