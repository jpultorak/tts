from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm

from tts import config, tokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    print(f"Root Directory: {ROOT_DIR}")

    print("Loading/Downloading LJSpeech Dataset...")

    raw_data = ROOT_DIR / "data_raw"

    # We use torchaudio just for the download logic.
    torchaudio.datasets.LJSPEECH(root=raw_data, download=True)

    dataset_root = config.ROOT_DIR / "data_raw" / "LJSpeech-1.1"
    wavs_dir = dataset_root / "wavs"
    meta_path = dataset_root / "metadata.csv"

    print("Loading EnCodec...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(config.ENCODEC_BANDWIDTH)
    model.eval()

    print("Loading Metadata & Building Vocab...")
    df = pd.read_csv(
        meta_path, sep="|", header=None, names=["ID", "Text", "Normalized"], quoting=3
    )

    tok = tokenizer.Tokenizer(vocab_path=config.VOCAB_PATH)
    all_texts = df["Normalized"].tolist()
    tok.build_from_text(all_texts)

    output_dir = config.ROOT_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing files to {output_dir}...")
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["ID"]
        text_norm = row["Normalized"]
        wav_path = wavs_dir / f"{file_id}.wav"

        try:
            wav_np, sr = sf.read(wav_path)
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.ndim == 2:
                wav = wav.transpose(0, 1).unsqueeze(0)

            # Resample 22k -> 24k
            wav = convert_audio(wav, sr, 24000, 1)

            # Encode (Audio -> Tokens)
            with torch.no_grad():
                encoded_frames = model.encode(wav)
                # Extract Layer 0 only
                codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
                audio_tokens = codes[0, 0, :]

            phn_ids = tok.encode(text_norm)
            phn_tokens = torch.tensor(phn_ids)

            torch.save(
                {
                    "audio_tokens": audio_tokens.clone(),
                    "phonemes": phn_tokens.clone(),
                    "text": text_norm,
                    "file_id": file_id,
                },
                output_dir / f"{file_id}.pt",
            )

            success_count += 1

        except Exception as e:
            print(f"\n⚠️ Error processing {file_id}: {e}")
            fail_count += 1
            continue

    print("\n------------------------------------------------")
    print("Processing done")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
