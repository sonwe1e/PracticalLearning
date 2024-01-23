import torch
import numpy as np
from tqdm import tqdm
from sampler import KEulerSampler
import utils
from diffusion import Diffusion
from encoder import Encoder
from decoder import Decoder
from clip import tokenize, load


class Generator:
    def __init__(
        self,
    ):
        super(Generator, self).__init__()
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.clip, self.transforms = load("ViT-B/32", device=self.device)
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.diffusion = Diffusion().to(self.device)
        self.sampler = None
        self.clip.eval()
        self.encoder.eval()
        self.decoder.eval()

    def generate(
        self,
        prompts,
        uncond_prompts=None,
        input_images=None,
        strength=0.8,
        do_cfg=False,
        cfg_scale=7.5,
        height=512,
        width=512,
        sampler="k_euler",
        n_inference_steps=50,
        models={},
        seed=None,
    ):
        """
        Function invoked when calling the pipeline for generation.
        Args:
            prompts (`List[str]`):
                The prompts to guide the image generation.
            uncond_prompts (`List[str]`, *optional*, defaults to `[""] * len(prompts)`):
                The prompts not to guide the image generation. Ignored when not using guidance (i.e. ignored if
                `do_cfg` is False).
            input_images (List[Union[`PIL.Image.Image`, str]]):
                Images which are served as the starting point for the image generation.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `input_images`. Must be between 0 and 1.
                `input_images` will be used as a starting point, adding more noise to it the larger the `strength`.
                The number of denoising steps depends on the amount of noise initially added. When `strength` is 1,
                added noise will be maximum and the denoising process will run for the full number of iterations
                specified in `n_inference_steps`. A value of 1, therefore, essentially ignores `input_images`.
            do_cfg (`bool`, *optional*, defaults to True):
                Enable [classifier-free guidance](https://arxiv.org/abs/2207.12598).
            cfg_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale of classifier-free guidance. Ignored when it is disabled (i.e. ignored if
                `do_cfg` is False). Higher guidance scale encourages to generate images that are closely linked
                to the text `prompt`, usually at the expense of lower image quality.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image. Ignored when `input_images` are provided.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image. Ignored when `input_images` are provided.
            sampler (`str`, *optional*, defaults to "k_lms"):
                A sampler to be used to denoise the encoded image latents. Can be one of `"k_lms"`, `"k_euler"`,
                or `"k_euler_ancestral"`.
            n_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            models (`Dict[str, torch.nn.Module]`, *optional*):
                Preloaded models. If some or all models are not provided, they will be loaded dynamically.
            seed (`int`, *optional*):
                A seed to make generation deterministic.
            device (`str` or `torch.device`, *optional*):
                PyTorch device which the image generation happens. If not provided, 'cuda' or 'cpu' will be used.
            idle_device (`str` or `torch.device`, *optional*):
                PyTorch device which the models no longer in use are moved to.
        Returns:
            `List[PIL.Image.Image]`:
                The generated images.
        Note:
            This docstring is heavily copied from huggingface/diffusers.
        """
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts must be a non-empty list")

        if uncond_prompts and isinstance(uncond_prompts, list):
            raise ValueError("uncond_prompts must be a list if provided")
        uncond_prompts = uncond_prompts or [""] * len(prompts)
        if input_images and isinstance(input_images, list):
            raise ValueError("input_images must be a list if provided")

        if input_images and len(input_images) != len(prompts):
            raise ValueError(
                "input_images must have the same length as prompts if provided"
            )

        if not 0 < strength < 1:
            raise ValueError("strength must be between 0 and 1")

        if height % 8 or width % 8:
            raise ValueError("height and width must be divisible by 8")

        if sampler == "k_lms":
            self.sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler":
            self.sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        else:
            raise ValueError(
                f"Sampler {sampler} not supported. Must be one of 'k_lms' or 'k_euler'."
            )
        with torch.no_grad():
            if do_cfg:
                cond_tokens = tokenize(prompts)
                uncond_tokens = tokenize(uncond_prompts)
                cond_latent = self.clip.encode_text(cond_tokens.to(self.device))
                uncond_latent = self.clip.encode_text(uncond_tokens.to(self.device))
                context = torch.cat([cond_latent, uncond_latent])
            else:
                cond_tokens = tokenize(prompts)
                context = self.clip.encode_text(cond_tokens.to(self.device))
                noise_shape = (len(prompts), 4, height, width)
            if input_images:
                pass

            else:
                latents = torch.randn(noise_shape, device=self.device)
                latents *= self.sampler.initial_scale

            time_steps = tqdm(self.sampler.timesteps)
            for i, time_step in enumerate(time_steps):
                time_embedding = utils.get_time_embedding(time_step).to(self.device)
                input_latents = latents * self.sampler.get_input_scale()
                if do_cfg:
                    input_latents = input_latents.repeat(2, 1, 1, 1)
                output = self.diffusion(input_latents, context, time_embedding)
                if do_cfg:
                    output_cond, output_uncond = output.chunk(2)
                    output = cfg_scale * output_cond + (1 - cfg_scale) * output_uncond

                latents = self.sampler.step(latents, output)

            images = self.decoder(latents)
        return images
