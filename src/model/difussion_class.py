from typing import Literal, Tuple, Optional
import math
from typing import Tuple, Sequence, Set, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.difussion_utils import *

class Diffusion(nn.Module):
    """
    Precomputes y expone tensores necesarios para DDPM:
      - betas, alphas, alphas_cumprod (alpha_bar), y sus raíces/inversas
      - q_sample(x0, t, eps): x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps
      - predict_x0(x_t, eps_pred, t)
      - posterior_mean_variance(x_t, x0_hat, t) -> mean, var, logvar
      - p_sample_step(...): un paso DDPM (estocástico)
    """
    def __init__(
        self,
        T: int = 1000,
        schedule: ScheduleKind = "linear",
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        cosine_s: float = 0.008,
        clamp_x0: bool = True,
        dynamic_threshold: Optional[float] = None , img_size=None):

        super().__init__()
        self.T = int(T)
        self.clamp_x0 = clamp_x0
        self.dynamic_threshold = dynamic_threshold
        self.img_size = img_size

        if schedule == "linear":
            betas = beta_schedule_linear(T, beta_min, beta_max)
        elif schedule == "cosine":
            betas = beta_schedule_cosine(T, s=cosine_s)
        else:
            raise ValueError(f"schedule desconocido: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Buffers para mover a device automáticamente con .to(device)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.register_buffer("alphas_cumprod_prev", F.pad(alphas_cumprod[:-1], (1,0), value=1.0), persistent=False)

        # Posterior q(x_{t-1} | x_t, x_0) varianza y coeficientes
        # \tilde{beta}_t = (1 - \bar{α}_{t-1}) / (1 - \bar{α}_t) * β_t
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # numerically stable log
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20), persistent=False)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)), persistent=False)
        # posterior mean coef: (√ᾱ_{t-1} β_t / (1-ᾱ_t)) x0 + (√α_t (1-ᾱ_{t-1})/(1-ᾱ_t)) x_t
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            persistent=False)

        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod),
            persistent=False)

    # --------- q(x_t | x_0) y utilidades ----------

    def sample_timesteps(self, batch_size: int, device=None) -> torch.Tensor:
        """
        Devuelve t ~ Uniform{1..T-1} (evita t=0 para la pérdida, aunque puedes incluirlo).
        """
        if device is None:
            device = self.betas.device
        return torch.randint(1, self.T, (batch_size,), device=device, dtype=torch.long)


    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Muestra x_t ~ q(x_t | x_0) a partir de ruido eps ~ N(0, I):
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omb = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omb * eps


    ## Main function for Training ###
    def loss_simple(
        self,
        model_eps_pred_fn,   # callable(x_t, t) -> eps_pred
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        """
        L_simple (MSE) entre ε y ε̂:
           E[ || ε - ε̂(x_t, t) ||^2 ]
        Permite pesos opcionales (p.ej., reweighting por t).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, eps=noise)
        eps_pred = model_eps_pred_fn(x_t, t)
        mse = (noise - eps_pred).pow(2).mean(dim=(1,2,3))
        if weight is not None:
            mse = mse * weight
        return mse.mean()


    ## Inference ###
    def posterior_mean_variance(
        self, x_t: torch.Tensor, x0_hat: torch.Tensor, t: torch.Tensor):
        """
        Devuelve mean, var, logvar de q(x_{t-1} | x_t, x0_hat).
        """
        coef1 = extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_hat + coef2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        logvar = extract(self.posterior_log_variance, t, x_t.shape)
        return mean, var, logvar

    @torch.no_grad()
    def predict_x0(self, x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reconstruye x0_hat desde x_t y epsilon_predicho:
            x0_hat = (x_t - sqrt(1-ᾱ_t) * eps_pred) / sqrt(ᾱ_t)
        """
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_omb = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_hat = (x_t - sqrt_omb * eps_pred) / (sqrt_ab + 1e-12)

        # Opcional: dynamic thresholding (Nichol & Dhariwal) para recortar outliers
        if self.dynamic_threshold is not None:
            s = self.dynamic_threshold
            amax = x0_hat.detach().abs().flatten(1).max(dim=1).values  # (B,)
            # evitamos dividir por 0
            amax = torch.maximum(amax, torch.tensor(1.0, device=x0_hat.device, dtype=x0_hat.dtype))
            x0_hat = (x0_hat.transpose(0, 1) / amax.clamp(min=s).unsqueeze(-1).unsqueeze(-1)).transpose(0, 1)
            x0_hat = x0_hat.clamp(-1, 1)
        elif self.clamp_x0:
            x0_hat = x0_hat.clamp(-1, 1)
        return x0_hat


    ## Metodo Clave para Inferencia ##
    @torch.no_grad()
    def p_sample_step(
        self,
        model_eps_pred_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eta: float = 1.0,
        use_ema_model: bool = True,
        clip_x0: Optional[bool] = None,
        noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Un paso de muestreo DDPM:
           1) eps_pred = model(x_t, t)
           2) x0_hat   = predict_x0(x_t, eps_pred, t)
           3) posterior mean/var -> mu, var
           4) x_{t-1} = mu + sigma_t * z (si t>0); si t=0, devolver mu
        - eta no afecta DDPM clásico (variancia fija). Lo dejamos para compatibilidad con DDIM.
        """
        if clip_x0 is None:
            clip_x0 = self.clamp_x0

        eps_pred = model_eps_pred_fn(x_t, t) # Predecimos el ruido con el modelo entrenado
        x0_hat = self.predict_x0(x_t, eps_pred, t) # Reconstruimos imagen
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        mean, var, logvar = self.posterior_mean_variance(x_t, x0_hat, t) # Calculamos momentos posteriores

        nonzero_mask = (t > 0).float().reshape((x_t.shape[0],) + (1,) * (x_t.ndim - 1))
        if noise is None:
            noise = torch.randn_like(x_t)
        return mean + nonzero_mask * torch.exp(0.5 * logvar) * noise

    @torch.no_grad()
    def p_sample_step_ddim(
        self,
        model_eps_pred_fn,           # callable(x_t, t) -> eps_pred
        x_t: torch.Tensor,
        t: torch.Tensor,             # (B,) índice actual
        t_prev: torch.Tensor,        # (B,) índice anterior del schedule (<= t)
        eta: float = 0.0,
        clip_x0: bool | None = None,
        noise: torch.Tensor | None = None):

        """
        Un paso DDIM: t -> t_prev. Si eta=0, trayecto determinista (prob.-flow ODE).
        Fórmula:
          x_{t'} = sqrt(ā_{t'}) x0_hat
                + sqrt(1 - ā_{t'} - sigma^2) * dir
                + sigma * z
          dir = (x_t - sqrt(ā_t) x0_hat) / sqrt(1 - ā_t)
          sigma = eta * sqrt((1 - ā_{t'})/(1 - ā_t)) * sqrt(1 - ā_t/ā_{t'})
        """

        if clip_x0 is None:
            clip_x0 = self.clamp_x0
        if noise is None:
            noise = torch.randn_like(x_t)

        # ᾱ_t y ᾱ_{t'}
        a_t      = extract(self.alphas_cumprod,       t,      x_t.shape)
        a_t_prev = extract(self.alphas_cumprod, t_prev,      x_t.shape)

        # ε̂ y x0_hat
        eps_pred = model_eps_pred_fn(x_t, t)
        x0_hat   = self.predict_x0(x_t, eps_pred, t)
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        # dirección y sigma
        dir_xt = (x_t - torch.sqrt(a_t) * x0_hat) / torch.sqrt(1.0 - a_t + 1e-12)
        sigma  = eta * torch.sqrt((1.0 - a_t_prev) / (1.0 - a_t + 1e-12)) \
                      * torch.sqrt(1.0 - a_t / (a_t_prev + 1e-12))

        # actualización DDIM
        mean   = torch.sqrt(a_t_prev) * x0_hat
        add    = torch.sqrt(torch.clamp(1.0 - a_t_prev - sigma**2, min=0.0)) * dir_xt
        x_prev = mean + add + sigma * noise
        return x_prev