import soundfile as sf
import torch
from encodec import EncodecModel

import tts.config as config
from tts.model import BabyValle
from tts.tokenizer import Tokenizer


class TTSInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"--> Initializing Inference on {self.device}")

        self.tokenizer = Tokenizer()

        print("--> Loading BabyValle...")
        self.model = BabyValle(
            vocab_size=2048, d_model=512, nhead=8, num_layers=6, max_len=4096
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("--> Loading EnCodec...")
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(config.ENCODEC_BANDWIDTH)
        self.codec.to(self.device)
        self.codec.eval()

    def generate_audio(
        self, text: str, output_path: str = "output.wav", max_new_tokens=750
    ):
        print(f"--> Generating for: '{text}'")
        phonemes = self.tokenizer.encode(text)[:-2]
        input_ids = phonemes + [config.SEP_TOKEN_ID]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        generated = []
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_tensor)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # [1, 1]

            token_id = next_token.item()
            if token_id == config.EOS_TOKEN_ID:
                break

            generated.append(token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

        print(f"--> Generated {len(generated)} audio codes.")
        self.save_wav(generated, output_path)

    def save_wav(self, codes_list, path):
        codes_tensor = (
            torch.tensor(codes_list).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            decoded_audio = self.codec.decode([(codes_tensor, None)])

        audio_arr = decoded_audio.squeeze().cpu().numpy()

        sf.write(path, audio_arr, 24000)
        print(f"--> Saved to {path}")
