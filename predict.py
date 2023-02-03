import subprocess
from typing import Optional

import numpy as np

from run import *
from utils.ptp_utils import aggregate_attention
from utils.vis_utils import show_image_relevance

from cog import BasePredictor, Path, Input, BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Output(BaseModel):
    output_image: Path
    explanation: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "-p", "/root/.cache/huggingface/diffusers"])
        subprocess.run(["mv", "models--CompVis--stable-diffusion-v1-4", "/root/.cache/huggingface/diffusers/"])
        self.stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

    def predict(
            self,
            prompt: str = Input(description="Text to Image prompt", default="A cat and a dog"),
            attend_and_excite: bool =Input(description="Apply Attend-and-Excite (If not this is simple Stable diffusion"
                                                       " and tokens_indices are ignored).", default=True),
            token_indices: str = Input(description="A comma separated list of indices (starting from 1) you wish to"
                                                   "attend and excite (Relevant only with attend_and_excite=True.",
                                       default='[2,5]'),
            explain: bool = Input(description="Show attention map for each token.", default=True),
            seed: int = Input(description="Random seed.", default=0),
    ) -> Output:
        config = RunConfig(prompt=str(prompt), token_indices=eval(str(token_indices)), run_standard_sd=not bool(attend_and_excite))

        g = torch.Generator('cuda').manual_seed(int(seed))
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                              model=self.stable,
                              controller=controller,
                              token_indices=config.token_indices,
                              seed=g,
                              config=config)
        output_path = "output_image.png"
        image.save(output_path)
        output = Output(output_image=Path(output_path))
        if bool(explain):
            explanation_path = "explanation.png"
            explanation = show_cross_attention(attention_store=controller,
                                                       prompt=config.prompt,
                                                       tokenizer=self.stable.tokenizer,
                                                       res=16,
                                                       from_where=("up", "down", "mid"),
                                                       indices_to_alter=config.token_indices,
                                                       orig_image=image)
            explanation.save(explanation_path)
            output.explanation = Path(explanation_path)
        return output


def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    return ptp_utils.view_images(np.stack(images, axis=0), display_image=False)
