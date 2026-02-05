import soundfile as sf
import torch
from encodec import EncodecModel

import tts.config as config
from tts.model import TtsModel
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

        print("--> Loading TtsModel...")
        self.model = TtsModel(
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

        phonemes = self.tokenizer.encode(text)
        input_ids = phonemes + [config.SEP_TOKEN_ID]
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        generated = []

        temperature = 0.8  # Allow it to breathe!
        top_k = 50
        top_p = 0.9
        rep_penalty = 1.2

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_tensor)

            next_token_logits = logits[:, -1, :]
            for token_in_set in set(generated[-20:]):
                if next_token_logits[:, token_in_set] < 0:
                    next_token_logits[:, token_in_set] *= rep_penalty
                else:
                    next_token_logits[:, token_in_set] /= rep_penalty

            next_token_logits = next_token_logits / temperature

            next_token_logits[:, 1025:] = -float("inf")

            v, _ = torch.topk(next_token_logits, top_k)
            out_of_k = next_token_logits < v[:, [-1]]
            next_token_logits[out_of_k] = -float("inf")

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = -float("inf")

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
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
