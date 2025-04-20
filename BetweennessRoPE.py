import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from einops import rearrange

class BetweennessRoPE(nn.Module):
    """
    RoPE implementation with betweenness-based positional adjustments.

    This approach measures the 'betweenness' of each token in semantic space
    and adjusts its positional encoding accordingly, allowing the model to
    develop a more interconnected understanding of token relationships.

    Note: The 'compute_betweenness' method calculates scores based on the input 'x'.
          When used in attention, consider whether to compute based on original input, Q, or K.
    Note: The 'apply_rope' method uses a loop, which might be a bottleneck for long sequences.
          Vectorization using gather could improve performance but adds complexity.
    """
    def __init__(self, dim, max_seq_len=2048, adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.adjustment_scale = adjustment_scale # Base scale for adjustment

        self.content_proj = nn.Linear(dim, dim) # Project input for semantic distance calculation

        # Learnable gate to scale the betweenness adjustment
        self.betweenness_gate = nn.Parameter(torch.ones(1) * 0.5)

        self.register_buffer(
            "base_freqs",
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        )

    def precompute_freqs_cis(self, seq_len, device):
        """Compute frequency bands for RoPE"""
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, self.base_freqs.to(device)) # Shape (SeqLen, Dim/2)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        # Add batch and head dimensions for easier broadcasting later: Shape (1, SeqLen, 1, Dim/2)
        return freqs_cos.unsqueeze(0).unsqueeze(2), freqs_sin.unsqueeze(0).unsqueeze(2)

    def compute_betweenness(self, x):
        """
        Compute betweenness scores for each token in the sequence.
        Input x shape: (Batch, SeqLen, Heads, Dim) or (Batch*Heads, SeqLen, Dim)
        Output betweenness shape: (Batch, SeqLen, Heads)
        """
        shape = x.shape
        orig_dim = x.dim()
        heads = 1 # Default if input is not 4D

        if orig_dim == 4:
            batch, seq_len, heads, dim = x.shape
            # Reshape to (Batch*Heads, SeqLen, Dim) for easier processing
            x = x.transpose(1, 2).reshape(batch * heads, seq_len, dim)
        elif orig_dim == 3: # (Batch*Heads, SeqLen, Dim)
             batch_heads, seq_len, dim = x.shape
             # We need batch size separately later if orig_dim was 4
             # This assumes batch_heads is a multiple of heads if orig_dim was 4
             # If orig_dim was 3 (e.g. synthetic test), batch = batch_heads
             batch = batch_heads # Placeholder, corrected later if needed
        else: # (Batch, SeqLen, Dim)
            batch, seq_len, dim = x.shape[:3]


        content = self.content_proj(x) # Project to semantic space: (B*H, S, D) or (B, S, D)

        token_i = content.unsqueeze(2) # (B*H, S, 1, D)
        token_j = content.unsqueeze(1) # (B*H, 1, S, D)
        distances = torch.norm(token_i - token_j, dim=-1) # (B*H, S, S)

        # Initialize betweenness scores: (B*H, S) or (B, S)
        betweenness = torch.zeros(x.shape[0], seq_len, device=x.device)

        # Calculate betweenness for intermediate tokens
        if seq_len > 2:
            for i in range(seq_len - 2):
                direct = distances[:, i, i+2] # (B*H,)
                path = distances[:, i, i+1] + distances[:, i+1, i+2] # (B*H,)

                # Score = 1 if path == direct, 0 if path >> direct
                between_score = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
                betweenness[:, i+1] += between_score

            # Normalize scores (optional, helps keep adjustments bounded)
            # Normalization factor depends only on sequence length
            betweenness = betweenness / (seq_len - 2) # Normalize by max possible paths counted

        # Reshape back if input was 4D
        if orig_dim == 4:
            # We know the original batch and heads from the start
            batch_orig, _, heads_orig, _ = shape
            # Output shape: (Batch, Heads, SeqLen) -> (Batch, SeqLen, Heads)
            betweenness = betweenness.view(batch_orig, heads_orig, seq_len).transpose(1, 2)
        elif orig_dim == 3 and heads > 1: # Handle case where input was (B*H, S, D)
             # This requires knowing heads somehow, maybe pass it? For now, assume B=B*H
             pass # Cannot reshape back without knowing original B and H
        # If orig_dim was 2 or 3 (B, S, D), shape is already (B, S) - needs expansion
        elif orig_dim <= 3:
             betweenness = betweenness.unsqueeze(-1) # Add head dim: (B, S, 1)


        return betweenness # Shape (B, S, H) or (B, S, 1)

    def apply_rope(self, x, freqs_cos, freqs_sin, betweenness):
        """
        Apply RoPE rotation with betweenness-adjusted positions.
        x shape: (Batch, SeqLen, Heads, Dim)
        freqs_cos/sin shape: (1, MaxSeqLen, 1, Dim/2)
        betweenness shape: (Batch, SeqLen, Heads)
        """
        batch, seq_len, heads, dim = x.shape
        device = x.device
        assert dim == self.dim, f"Input dim {dim} doesn't match module dim {self.dim}"
        assert betweenness.shape == (batch, seq_len, heads), f"Betweenness shape mismatch: {betweenness.shape}"
        assert freqs_cos.shape[-1] == dim // 2, "Freqs shape mismatch"

        # Calculate position adjustments using the learnable gate
        # Shape: (B, S, H) -> (B, S, H, 1) for broadcasting with frac
        pos_adjustments = (self.betweenness_gate * (betweenness - 0.5) * self.adjustment_scale).unsqueeze(-1)

        x_out = torch.zeros_like(x, dtype=torch.float)

        # TODO: Potential optimization: Vectorize this loop using torch.gather
        for i in range(seq_len):
            # Calculate adjusted position: scalar i + (B, 1, H, 1) -> (B, 1, H, 1)
            # We slice betweenness to get (B, 1, H) then unsqueeze
            current_adjust = (self.betweenness_gate * (betweenness[:, i:i+1, :] - 0.5) * self.adjustment_scale)
            adj_pos = torch.clamp(i + current_adjust, 0, self.max_seq_len - 1) # Clamp to valid range

            # Get integer indices and interpolation fraction
            idx_lo = torch.floor(adj_pos).long() # Shape (B, 1, H)
            idx_hi = torch.ceil(adj_pos).long()  # Shape (B, 1, H)
            frac = (adj_pos - idx_lo).unsqueeze(-1) # Shape (B, 1, H, 1)

            # Gather frequencies for interpolation
            # Need to index freqs_cos/sin which are (1, MaxSeqLen, 1, Dim/2)
            # idx_lo/hi are (B, 1, H), need (B, NumIndices, H, Dim/2) for gather
            # Expand indices to match freq dims for gather
            idx_lo_expanded = idx_lo.unsqueeze(-1).expand(-1, -1, -1, dim // 2) # (B, 1, H, D/2)
            idx_hi_expanded = idx_hi.unsqueeze(-1).expand(-1, -1, -1, dim // 2) # (B, 1, H, D/2)

            # Gather requires freqs to have batch dim matching index batch dim
            # Expand freqs batch dim: (1, MaxSeqLen, 1, D/2) -> (B, MaxSeqLen, 1, D/2)
            freqs_cos_batch = freqs_cos.expand(batch, -1, -1, -1)
            freqs_sin_batch = freqs_sin.expand(batch, -1, -1, -1)

            # Gather along SeqLen dimension (dim=1)
            cos_lo = torch.gather(freqs_cos_batch, 1, idx_lo_expanded.transpose(1,2)).transpose(1,2) # (B, 1, H, D/2)
            cos_hi = torch.gather(freqs_cos_batch, 1, idx_hi_expanded.transpose(1,2)).transpose(1,2) # (B, 1, H, D/2)
            sin_lo = torch.gather(freqs_sin_batch, 1, idx_lo_expanded.transpose(1,2)).transpose(1,2) # (B, 1, H, D/2)
            sin_hi = torch.gather(freqs_sin_batch, 1, idx_hi_expanded.transpose(1,2)).transpose(1,2) # (B, 1, H, D/2)


            # Interpolate frequencies: (B, 1, H, D/2)
            cos_interp = (1 - frac) * cos_lo + frac * cos_hi
            sin_interp = (1 - frac) * sin_lo + frac * sin_hi

            # Squeeze the sequence dimension (dim 1) -> (B, H, D/2)
            cos_interp = cos_interp.squeeze(1)
            sin_interp = sin_interp.squeeze(1)

            # Apply rotation to the current token x[:, i, :, :] which has shape (B, H, D)
            x_i = x[:, i, :, :] # (B, H, D)
            x_even = x_i[..., 0::2] # (B, H, D/2)
            x_odd = x_i[..., 1::2]  # (B, H, D/2)

            x_out[:, i, :, 0::2] = x_even * cos_interp - x_odd * sin_interp
            x_out[:, i, :, 1::2] = x_odd * cos_interp + x_even * sin_interp

        return x_out.type_as(x)

    # Removed forward method - logic moved to Attention layer


class BetweennessAttention(nn.Module):
    """
    Attention mechanism using betweenness-adjusted RoPE embeddings.
    Computes betweenness once based on Q projection.
    """
    def __init__(self, dim, heads, dim_head=64, rope_max_seq_len=2048, rope_adjustment_scale=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.q_proj = nn.Linear(dim, inner_dim)
        self.k_proj = nn.Linear(dim, inner_dim)
        self.v_proj = nn.Linear(dim, inner_dim)
        self.output = nn.Linear(inner_dim, dim)

        # Instantiate RoPE module
        self.rope = BetweennessRoPE(dim_head, max_seq_len=rope_max_seq_len, adjustment_scale=rope_adjustment_scale)

        # Precompute frequencies only once if max_seq_len is fixed
        # Note: If handling variable sequence lengths up to max_seq_len,
        # precomputation might need adjustment or slicing in forward pass.
        # Here, we precompute for the maximum possible length.
        self.freqs_cos, self.freqs_sin = self.rope.precompute_freqs_cis(rope_max_seq_len, device='cpu') # Compute on CPU initially

    def _to_device(self, device):
        # Helper to move precomputed freqs to the correct device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
        if self.freqs_sin.device != device:
            self.freqs_sin = self.freqs_sin.to(device)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape # Use input seq_len
        device = x.device
        self._to_device(device) # Ensure freqs are on the right device

        q = self.q_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        k = self.k_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)
        v = self.v_proj(x).reshape(batch, seq_len, self.heads, self.dim_head)

        # --- Betweenness RoPE Application ---
        # 1. Compute betweenness scores (e.g., based on Q)
        # Input to compute_betweenness: (B, S, H, D)
        # Output betweenness: (B, S, H)
        betweenness = self.rope.compute_betweenness(q)

        # 2. Get precomputed frequencies for the current sequence length
        # Slice the precomputed freqs: (1, MaxSeqLen, 1, D/2) -> (1, SeqLen, 1, D/2)
        current_freqs_cos = self.freqs_cos[:, :seq_len, :, :]
        current_freqs_sin = self.freqs_sin[:, :seq_len, :, :]

        # 3. Apply RoPE using the computed betweenness and sliced frequencies
        q_rope = self.rope.apply_rope(q, current_freqs_cos, current_freqs_sin, betweenness)
        k_rope = self.rope.apply_rope(k, current_freqs_cos, current_freqs_sin, betweenness) # Reuse betweenness
        # --- End Betweenness RoPE ---

        # --- Standard Attention Calculation ---
        q_rope = q_rope.transpose(1, 2) # (B, H, S, D)
        k_rope = k_rope.transpose(1, 2) # (B, H, S, D)
        v = v.transpose(1, 2)      # (B, H, S, D)

        scores = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(self.dim_head)

        if mask is not None:
            # Ensure mask has compatible shape (e.g., B, 1, S, S)
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1) # (B, H, S, S)

        output = torch.matmul(weights, v) # (B, H, S, D)

        output = output.transpose(1, 2).reshape(batch, seq_len, -1) # (B, S, H*D)
        return self.output(output)


def test_betweenness_rope():
    torch.manual_seed(42)

    dim = 64
    heads = 4
    dim_head = dim // heads
    seq_len = 10
    batch_size = 2
    max_seq_len_test = 20 # For RoPE module
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(batch_size, seq_len, dim).to(device)

    # --- Test RoPE module components directly ---
    print("--- Direct RoPE Module Test ---")
    rope_module = BetweennessRoPE(dim=dim_head, max_seq_len=max_seq_len_test).to(device)
    # Reshape x to match expected input for compute_betweenness when called directly
    # (B, S, D) -> (B, S, H, D_head)
    x_reshaped_for_rope = x.reshape(batch_size, seq_len, heads, dim_head)
    betweenness_direct = rope_module.compute_betweenness(x_reshaped_for_rope)

    print(f"Direct Betweenness shape: {betweenness_direct.shape}") # Expect (B, S, H)
    print(f"Direct Betweenness scores (Batch 0, Head 0): \n{betweenness_direct[0, :, 0]}")

    # --- Test Attention module which uses RoPE internally ---
    print("\n--- Attention Module Test ---")
    attn = BetweennessAttention(dim=dim, heads=heads, dim_head=dim_head, rope_max_seq_len=max_seq_len_test).to(device)

    # --- Manually perform Q/K projection and RoPE application for inspection ---
    print("\n--- Manual Q/K RoPE Application within Attention Test ---")
    attn._to_device(device) # Ensure frequencies are on the correct device

    q = attn.q_proj(x).reshape(batch_size, seq_len, heads, dim_head)
    k = attn.k_proj(x).reshape(batch_size, seq_len, heads, dim_head)
    print(f"Original Q shape: {q.shape}") # Expect (B, S, H, D_head)
    print(f"Original K shape: {k.shape}") # Expect (B, S, H, D_head)

    # 1. Compute betweenness (based on Q, as in attn.forward)
    betweenness_attn = attn.rope.compute_betweenness(q)
    print(f"Attention Betweenness shape: {betweenness_attn.shape}") # Expect (B, S, H)
    print(f"Attention Betweenness scores (Batch 0, Head 0): \n{betweenness_attn[0, :, 0]}")

    # 2. Get precomputed frequencies for the current sequence length
    current_freqs_cos = attn.freqs_cos[:, :seq_len, :, :]
    current_freqs_sin = attn.freqs_sin[:, :seq_len, :, :]
    print(f"Sliced Freqs Cos shape: {current_freqs_cos.shape}") # Expect (1, S, 1, D_head/2)

    # 3. Apply RoPE using the computed betweenness and sliced frequencies
    q_rope = attn.rope.apply_rope(q, current_freqs_cos, current_freqs_sin, betweenness_attn)
    k_rope = attn.rope.apply_rope(k, current_freqs_cos, current_freqs_sin, betweenness_attn) # Reuse betweenness
    print(f"Q after RoPE shape: {q_rope.shape}") # Expect (B, S, H, D_head)
    print(f"K after RoPE shape: {k_rope.shape}") # Expect (B, S, H, D_head)

    # Optional: Print some values to see the effect
    print(f"Original Q (Batch 0, Pos 0, Head 0, First 5 vals): \n{q[0, 0, 0, :5]}")
    print(f"RoPE Q (Batch 0, Pos 0, Head 0, First 5 vals): \n{q_rope[0, 0, 0, :5]}")
    print(f"Original K (Batch 0, Pos 0, Head 0, First 5 vals): \n{k[0, 0, 0, :5]}")
    print(f"RoPE K (Batch 0, Pos 0, Head 0, First 5 vals): \n{k_rope[0, 0, 0, :5]}")
    # --- End Manual Q/K RoPE ---

    # Run the full attention forward pass
    output = attn(x)
    print(f"\nAttention Final Output shape: {output.shape}") # Expect (B, S, Dim)

    # Visualize the betweenness calculated directly for the first batch item
    mean_betweenness_scores_direct = betweenness_direct[0].mean(dim=-1).detach().cpu().numpy()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(seq_len), mean_betweenness_scores_direct)
    plt.title("Mean Betweenness (Direct Calc, B0)")
    plt.xlabel("Token Position")
    plt.ylabel("Mean Score")

    # Visualize the betweenness calculated inside attention for the first batch item
    mean_betweenness_scores_attn = betweenness_attn[0].mean(dim=-1).detach().cpu().numpy()
    plt.subplot(1, 2, 2)
    plt.bar(range(seq_len), mean_betweenness_scores_attn)
    plt.title("Mean Betweenness (Attention Calc on Q, B0)")
    plt.xlabel("Token Position")
    plt.ylabel("Mean Score")

    plt.tight_layout()
    plt.show()


def test_synthetic_betweenness():
    print("\n--- Synthetic Betweenness Test ---")
    dim = 64
    seq_len = 5
    max_seq_len_test = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create synthetic data where token 1 is between 0&2, token 3 is between 2&4
    x = torch.zeros(1, seq_len, dim).to(device)
    x[0, 0] = torch.tensor([1.0, 0.0, 0.0] + [0.0] * (dim - 3))
    x[0, 1] = torch.tensor([0.5, 0.5, 0.0] + [0.0] * (dim - 3)) # Midpoint of 0 and 2
    x[0, 2] = torch.tensor([0.0, 1.0, 0.0] + [0.0] * (dim - 3))
    x[0, 3] = torch.tensor([0.0, 0.5, 0.5] + [0.0] * (dim - 3)) # Midpoint of 2 and 4
    x[0, 4] = torch.tensor([0.0, 0.0, 1.0] + [0.0] * (dim - 3))

    # Use RoPE module with identity projection for testing the core logic
    rope = BetweennessRoPE(dim, max_seq_len=max_seq_len_test).to(device)
    with torch.no_grad():
        # Ensure identity projection is on the correct device
        rope.content_proj.weight.copy_(torch.eye(dim, device=device))
        rope.content_proj.bias.fill_(0.0)

    # Compute betweenness (input needs head dim if module expects 4D, add dummy head)
    # Input shape (B, S, D) -> (B, S, 1, D)
    # Output shape (B, S, 1)
    betweenness = rope.compute_betweenness(x.unsqueeze(2))

    print(f"Synthetic example betweenness scores (Shape: {betweenness.shape}):")
    # Squeeze head dim for printing/plotting: (B, S, 1) -> (B, S)
    print(betweenness[0].squeeze(-1))

    plt.figure(figsize=(10, 5))
    plt.bar(range(seq_len), betweenness[0].squeeze(-1).detach().cpu().numpy())
    plt.title("Betweenness Scores - Synthetic Example")
    plt.xlabel("Token Position")
    plt.ylabel("Betweenness Score")
    plt.xticks(range(seq_len))
    plt.show()


# if __name__ == "__main__":
#     test_betweenness_rope()
#     test_synthetic_betweenness()