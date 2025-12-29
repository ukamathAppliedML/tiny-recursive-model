"""
Tiny Recursive Model Architecture
==================================

This module implements the core TRM architecture from
"Less is More: Recursive Reasoning with Tiny Networks".

The key components are:
- RMSNorm: Root Mean Square normalization
- RotaryEmbedding: Rotary Position Embeddings (RoPE)
- SwiGLU: Gated Linear Unit with Swish activation
- MLPMixerBlock: Token mixing via MLP (alternative to attention)
- MultiHeadAttention: Standard attention with RoPE
- TRMBlock: Single transformer/mixer block
- TinyRecursiveNetwork: The 2-layer recursive network
- TinyRecursiveModel: Complete model with embeddings and heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from .config import TRMConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable than LayerNorm for transformers, as used in LLaMA and other
    modern architectures. Normalizes by RMS without centering.
    
    Parameters
    ----------
    dim : int
        Feature dimension
    eps : float
        Small constant for numerical stability
    
    References
    ----------
    Zhang & Sennrich, 2019: "Root Mean Square Layer Normalization"
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Encodes position information by rotating queries and keys in the
    complex plane. More flexible than learned positional embeddings.
    
    Parameters
    ----------
    dim : int
        Head dimension (must be even)
    max_seq_len : int
        Maximum sequence length to precompute
    base : float
        Base for frequency computation
    
    References
    ----------
    Su et al., 2024: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, seq_len, dim]
        offset : int
            Position offset (for handling long sequences)
        
        Returns
        -------
        torch.Tensor
            Rotated tensor with same shape as input
        """
        seq_len = x.shape[1]
        
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        # Split into even/odd indices
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated


class SwiGLU(nn.Module):
    """SwiGLU Activation Function.
    
    Gated Linear Unit with Swish (SiLU) activation:
    SwiGLU(x) = Swish(xW1) ⊙ (xW2)
    
    More expressive than ReLU/GELU, commonly used in modern LLMs.
    
    Parameters
    ----------
    in_features : int
        Input dimension
    hidden_features : int
        Hidden dimension
    bias : bool
        Whether to use bias (paper uses no bias)
    
    References
    ----------
    Shazeer, 2020: "GLU Variants Improve Transformer"
    """
    
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)  # Value
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)  # Output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))  # Swish activation
        value = self.w2(x)
        return self.w3(gate * value)


class MLPMixerBlock(nn.Module):
    """MLP-Mixer style token mixing.
    
    Replaces attention with an MLP that operates across the sequence
    dimension. Efficient when context_length <= hidden_dim.
    
    The paper found this works better than attention for Sudoku!
    
    Parameters
    ----------
    config : TRMConfig
        Model configuration
    
    References
    ----------
    Tolstikhin et al., 2021: "MLP-Mixer: An All-MLP Architecture for Vision"
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(config.context_length, config.context_length * 4),
            nn.GELU(),
            nn.Linear(config.context_length * 4, config.context_length)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # Transpose: mix across token dimension
        x = x.transpose(1, 2)
        x = self.token_mixer(x)
        x = x.transpose(1, 2)
        return residual + x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE.
    
    Standard transformer attention with rotary position embeddings.
    Used when config.use_attention=True.
    
    Parameters
    ----------
    config : TRMConfig
        Model configuration
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        # No bias as per paper
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.rope = RotaryEmbedding(self.head_dim, config.context_length)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        q = self.rope(q.reshape(B * self.num_heads, L, self.head_dim))
        k = self.rope(k.reshape(B * self.num_heads, L, self.head_dim))
        q = q.reshape(B, self.num_heads, L, self.head_dim)
        k = k.reshape(B, self.num_heads, L, self.head_dim)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TRMBlock(nn.Module):
    """Single TRM Transformer/Mixer Block.
    
    Architecture:
        x -> Norm -> Token Mixer -> + -> Norm -> SwiGLU MLP -> +
             └──────────────────────┘    └──────────────────────┘
    
    Parameters
    ----------
    config : TRMConfig
        Model configuration
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)
        
        # Token mixing
        if config.use_attention:
            self.token_mixer = MultiHeadAttention(config)
            self.needs_norm_for_mixer = True
        else:
            self.token_mixer = MLPMixerBlock(config)
            self.needs_norm_for_mixer = False  # MLPMixerBlock has its own norm
        
        # Channel MLP
        hidden_dim = int(config.hidden_dim * config.mlp_ratio)
        self.channel_mlp = SwiGLU(config.hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        if self.needs_norm_for_mixer:
            x = x + self.token_mixer(self.norm1(x))
        else:
            x = self.token_mixer(x)
        
        # Channel MLP
        x = x + self.channel_mlp(self.norm2(x))
        return x


class TinyRecursiveNetwork(nn.Module):
    """The tiny network f that is recursively applied.
    
    This is a stack of TRMBlocks (typically just 2 layers).
    The same network is used for both:
    - z updates: z = f(x + y + z)
    - y updates: y = f(y + z)
    
    Parameters
    ----------
    config : TRMConfig
        Model configuration
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        
        self.blocks = nn.ModuleList([
            TRMBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class TinyRecursiveModel(nn.Module):
    """Complete Tiny Recursive Model.
    
    The full model including:
    - Token embedding
    - Learnable initial states (y_init, z_init)
    - Recursive network
    - Output and halting heads
    
    Parameters
    ----------
    config : TRMConfig
        Model configuration
    
    Examples
    --------
    >>> config = TRMConfig(vocab_size=10, context_length=81)
    >>> model = TinyRecursiveModel(config)
    >>> 
    >>> # Single supervision step
    >>> x = torch.randint(0, 10, (4, 81))
    >>> (y, z), logits, halt = model(x)
    >>> 
    >>> # Multiple steps with state carry
    >>> for step in range(16):
    ...     (y, z), logits, halt = model(x, y, z)
    >>> 
    >>> # Inference with all steps
    >>> predictions = model.predict(x)
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Learnable initial states
        self.y_init = nn.Parameter(
            torch.randn(1, config.context_length, config.hidden_dim) * 0.02
        )
        self.z_init = nn.Parameter(
            torch.randn(1, config.context_length, config.hidden_dim) * 0.02
        )
        
        # The single tiny network
        self.net = TinyRecursiveNetwork(config)
        
        # Output heads
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.q_head = nn.Linear(config.hidden_dim, 1, bias=False)  # Halting
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform n latent recursions then one answer refinement.
        
        Algorithm:
            for i in range(n):
                z = net(x + y + z)   # Update reasoning
            y = net(y + z)           # Refine answer (no x!)
        
        Parameters
        ----------
        x : torch.Tensor
            Embedded input [batch, seq_len, hidden_dim]
        y : torch.Tensor
            Current answer embedding
        z : torch.Tensor
            Current latent state
        n : int, optional
            Number of recursions (default: config.n_recursions)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated (y, z)
        """
        n = n or self.config.n_recursions
        
        # Update latent n times
        for _ in range(n):
            z = self.net(x + y + z)
        
        # Refine answer once (no x!)
        y = self.net(y + z)
        
        return y, z
    
    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: Optional[int] = None,
        T: Optional[int] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Deep recursion with gradient-free pre-steps.
        
        Key insight: Do T-1 recursions without gradients, then 1 with gradients.
        This emulates very deep networks without the memory cost.
        
        Parameters
        ----------
        x : torch.Tensor
            Embedded input
        y : torch.Tensor
            Current answer embedding
        z : torch.Tensor
            Current latent state
        n : int, optional
            Recursions per step
        T : int, optional
            Total steps (T-1 are gradient-free)
        
        Returns
        -------
        Tuple[Tuple[Tensor, Tensor], Tensor, Tensor]
            (y, z): Updated states (detached)
            logits: Output predictions [batch, seq_len, vocab_size]
            q_hat: Halting probability logit [batch]
        """
        n = n or self.config.n_recursions
        T = T or self.config.T_steps
        
        # T-1 gradient-free recursions
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        
        # 1 recursion with gradients
        y, z = self.latent_recursion(x, y, z, n)
        
        # Compute outputs
        logits = self.output_head(y)
        q_hat = self.q_head(y).mean(dim=(1, 2))
        
        return (y.detach(), z.detach()), logits, q_hat
    
    def forward(
        self,
        input_tokens: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass for one supervision step.
        
        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token IDs [batch, seq_len]
        y : torch.Tensor, optional
            Previous answer embedding (None for first step)
        z : torch.Tensor, optional
            Previous latent state (None for first step)
        
        Returns
        -------
        Tuple[Tuple[Tensor, Tensor], Tensor, Tensor]
            (y, z): Updated states for next step
            logits: [batch, seq_len, vocab_size]
            q_hat: Halting probability [batch]
        """
        batch_size = input_tokens.shape[0]
        
        # Embed input
        x = self.token_embedding(input_tokens)
        
        # Initialize states if needed
        if y is None:
            y = self.y_init.expand(batch_size, -1, -1)
        if z is None:
            z = self.z_init.expand(batch_size, -1, -1)
        
        return self.deep_recursion(x, y, z)
    
    @torch.no_grad()
    def predict(
        self,
        input_tokens: torch.Tensor,
        n_supervision: Optional[int] = None
    ) -> torch.Tensor:
        """Generate predictions with full supervision steps.
        
        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token IDs [batch, seq_len]
        n_supervision : int, optional
            Number of supervision steps (default: config.n_supervision)
        
        Returns
        -------
        torch.Tensor
            Predicted token IDs [batch, seq_len]
        """
        n_supervision = n_supervision or self.config.n_supervision
        
        y, z = None, None
        for _ in range(n_supervision):
            (y, z), logits, _ = self.forward(input_tokens, y, z)
        
        return logits.argmax(dim=-1)
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
