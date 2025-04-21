import torch
from typing import Optional, Dict, Any

class UniversalMask:
    """
    A unified masking solution for transformer models that works with any attention mechanism.
    
    This class provides a simple interface for creating, converting, combining and applying
    attention masks, with functionality for:
    - Causal masking (sequence can only attend to previous positions)
    - Padding masking (ignoring padded positions in sequences)
    - Head-specific masking (different patterns per attention head)
    - Visualization tools for debugging
    
    The class is designed to work seamlessly with torch.nn.functional.scaled_dot_product_attention
    and custom attention implementations.
    """
    
    @staticmethod
    def create(
        batch_size: int,
        seq_len: int,
        num_heads: Optional[int] = None,
        mask_type: str = "attention",
        **kwargs
    ) -> torch.Tensor:
        """
        Create a universal mask that works across different attention mechanisms.
        
        Args:
            batch_size (int): Batch size
            seq_len (int): Sequence length
            num_heads (int, optional): Number of attention heads
            mask_type (str): Type of mask to create ('attention', 'padding', 'causal', 'combined')
            **kwargs: Additional arguments
                - is_causal (bool): Whether to create a causal mask
                - padding_mask (Tensor): Boolean tensor where True indicates padding positions
                - device (torch.device): Device for the mask tensor
                - dtype (torch.dtype): Data type for the mask tensor
                - expand_dims (bool): Whether to expand dimensions for multi-head attention
        
        Returns:
            torch.Tensor: The created mask in appropriate format for attention operations
        """
        device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        is_causal = kwargs.get('is_causal', False)
        padding_mask = kwargs.get('padding_mask', None)
        dtype = kwargs.get('dtype', torch.float32)
        expand_dims = kwargs.get('expand_dims', True)
        
        if mask_type == "causal" or (mask_type == "combined" and is_causal):
            # Create a causal mask in canonical form (0 for attention, -inf for masking)
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
            mask = mask.masked_fill(torch.triu(torch.ones_like(mask), diagonal=1).bool(), float("-inf"))
        else:
            # Create an empty mask (all positions can attend)
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            if padding_mask.dim() == 2:  # [batch_size, seq_len]
                # Convert to proper format for broadcasting
                padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                
                if mask_type in ["padding", "combined"]:
                    # Start with the existing mask and add padding constraints
                    if mask.dim() == 2:  # [seq_len, seq_len]
                        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
                    
                    # Apply padding mask (-inf where padding tokens exist)
                    mask = mask.masked_fill(
                        padding_mask_expanded.expand(-1, -1, seq_len, -1), 
                        float("-inf")
                    )
        
        # Expand dimensions for attention heads if needed
        if expand_dims and num_heads is not None:
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                mask = mask.expand(batch_size, num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
                mask = mask.expand(-1, num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
        
        return mask

    @staticmethod
    def convert(
        mask: torch.Tensor,
        source_type: str = "bool",
        target_type: str = "attention"
    ) -> torch.Tensor:
        """
        Convert between different mask types.
        
        Args:
            mask (torch.Tensor): Mask tensor to convert
            source_type (str): Source mask type ('bool', 'attention', 'additive')
            target_type (str): Target mask type ('bool', 'attention', 'additive')
        
        Returns:
            torch.Tensor: Converted mask
        """
        if source_type == target_type:
            return mask
        
        if source_type == "bool" and target_type == "attention":
            # Convert from bool (True=masked) to attention format (0=attend, -inf=don't attend)
            attn_mask = torch.zeros_like(mask, dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(mask, float("-inf"))
            return attn_mask
            
        if source_type == "attention" and target_type == "bool":
            # Convert from attention format to bool (True=masked)
            return mask == float("-inf")
            
        # Add more conversions as needed
        
        return mask

    @staticmethod
    def combine(mask1: Optional[torch.Tensor], mask2: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Combine two masks.
        
        Args:
            mask1 (torch.Tensor): First mask in attention format
            mask2 (torch.Tensor): Second mask in attention format
        
        Returns:
            torch.Tensor: Combined mask
        """
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
            
        # For attention masks (-inf values), we can add them together
        # This works because -inf + x = -inf for any finite x
        return mask1 + mask2

    @staticmethod
    def visualize(mask: torch.Tensor, title: str = "Attention Mask") -> None:
        """
        Visualize a mask for debugging purposes.
        
        Args:
            mask (torch.Tensor): Mask tensor to visualize
            title (str): Title for the visualization
        """
        import matplotlib.pyplot as plt
        if mask.dim() == 4:
            mask_vis = mask[0, 0].cpu().detach().numpy()
        else:
            mask_vis = mask.cpu().detach().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_vis, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

    @staticmethod
    def get_sdpa_compatible_mask(
        mask: Optional[torch.Tensor], 
        batch_size: int, 
        num_heads: int,
        seq_len: int, 
        is_causal: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Get a mask compatible with torch.nn.functional.scaled_dot_product_attention.
        
        Args:
            mask (torch.Tensor, optional): Existing mask
            batch_size (int): Batch size
            num_heads (int): Number of attention heads
            seq_len (int): Sequence length
            is_causal (bool): Whether to use causal masking
            device (torch.device, optional): Device for the mask tensor
            
        Returns:
            dict: Dictionary with 'attn_mask' and 'is_causal' keys
        """
        if mask is not None:
            return {'attn_mask': mask, 'is_causal': None}
        elif is_causal:
            return {'attn_mask': None, 'is_causal': True}
        else:
            return {'attn_mask': None, 'is_causal': None}

class ASRMask:
    """
    Specialized masking solution for ASR (Automatic Speech Recognition) models.
    
    This class provides masking functionality specifically optimized for encoder-decoder
    ASR architectures where:
    - Audio encoders must process silence (zeros in mel spectrograms) without masking
    - Text decoders require strict causal masking to prevent "peeking ahead"
    - Cross-attention requires careful handling to prevent contamination
    
    Features:
    - Audio-specific masks (preserves silence as meaningful information)
    - Text-specific causal masks (with padding token support)
    - Cross-attention masks for encoder-decoder interfaces
    - Support for attention mechanisms used in modern ASR systems
    """
    
    @staticmethod
    def create_audio_encoder_mask(
        batch_size: int,
        seq_len: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        Create a mask for audio encoder self-attention.
        For audio encoders processing mel spectrograms, we typically don't want any masking
        since all time steps (including zeros/silence) contain meaningful information.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            device: Device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            None - Audio encoders typically shouldn't mask any positions including silence
        """
        # For audio encoders, we explicitly return None to indicate no masking
        # This ensures the model learns the importance of silence in speech
        return None
    
    @staticmethod
    def create_text_decoder_mask(
        batch_size: int,
        seq_len: int,
        num_heads: int,
        padding_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Create a mask for text decoder self-attention.
        For text decoders, we need causal masking (can't see future tokens) and
        padding masking (ignore pad tokens).
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            padding_mask: Boolean tensor where True indicates padding positions
            device: Device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            torch.Tensor: Properly formatted attention mask for text decoder
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create causal mask - absolutely essential for autoregressive text generation
        # Shape: [seq_len, seq_len]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), 
            diagonal=1
        )
        
        # Expand to [batch_size, num_heads, seq_len, seq_len]
        mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask shape: [batch_size, seq_len] - True for pad tokens
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            
            # Mask where tokens can't attend to pad tokens
            # Expand to [batch_size, 1, seq_len, seq_len]
            pad_attn_mask = padding_mask_expanded.expand(-1, -1, seq_len, -1)
            
            # Combine with causal mask
            # This ensures we respect both causality and padding constraints
            mask = mask.masked_fill(pad_attn_mask, float("-inf"))
        
        return mask
    
    @staticmethod
    def create_cross_attention_mask(
        batch_size: int,
        text_seq_len: int,
        audio_seq_len: int,
        num_heads: int,
        audio_padding_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        Create a mask for cross-attention from text decoder to audio encoder.
        
        In cross-attention:
        - Each text token should attend to all relevant audio frames
        - If audio was padded, we should mask those padded frames
        
        Args:
            batch_size: Batch size
            text_seq_len: Text sequence length (queries)
            audio_seq_len: Audio sequence length (keys/values)
            num_heads: Number of attention heads
            audio_padding_mask: Boolean mask where True indicates audio padding
            device: Device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            torch.Tensor or None: Cross-attention mask or None if no masking needed
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For cross-attention, typically we don't need causal masking
        # Each text token can attend to all audio frames
        
        # Only create a mask if we need to mask padded audio frames
        if audio_padding_mask is not None:
            # audio_padding_mask shape: [batch_size, audio_seq_len]
            # Expand to [batch_size, 1, 1, audio_seq_len]
            padding_mask_expanded = audio_padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Create cross-attention mask
            # Shape: [batch_size, num_heads, text_seq_len, audio_seq_len]
            cross_attn_mask = torch.zeros(
                (batch_size, num_heads, text_seq_len, audio_seq_len),
                device=device,
                dtype=dtype
            )
            
            # Apply audio padding mask
            # This ensures text tokens don't attend to padded audio frames
            cross_attn_mask = cross_attn_mask.masked_fill(
                padding_mask_expanded.expand(-1, num_heads, text_seq_len, -1),
                float("-inf")
            )
            
            return cross_attn_mask
        
        # No masking needed if audio wasn't padded
        return None
    
    @staticmethod
    def get_mask_for_module(
        module_type: str,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        is_causal: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        cross_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """
        Get appropriate mask for specific module types in an ASR model.
        
        Args:
            module_type: Type of module ('audio_encoder', 'text_decoder', 'cross_attention')
            batch_size: Batch size
            seq_len: Primary sequence length
            num_heads: Number of attention heads
            is_causal: Whether causal masking is needed
            padding_mask: Boolean tensor where True indicates padding positions
            cross_seq_len: Secondary sequence length (for cross-attention)
            device: Device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            torch.Tensor or None: Appropriate mask for the module or None
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if module_type == "audio_encoder":
            # No masking for audio encoder - ensure silence is processed
            return None
            
        elif module_type == "text_decoder":
            # Always use causal masking for text decoder
            return ASRMask.create_text_decoder_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                padding_mask=padding_mask,
                device=device,
                dtype=dtype
            )
            
        elif module_type == "cross_attention":
            if cross_seq_len is None:
                raise ValueError("cross_seq_len must be provided for cross_attention mask")
                
            # For cross-attention, text queries attend to audio keys/values
            return ASRMask.create_cross_attention_mask(
                batch_size=batch_size,
                text_seq_len=seq_len,
                audio_seq_len=cross_seq_len,
                num_heads=num_heads,
                audio_padding_mask=padding_mask,
                device=device,
                dtype=dtype
            )
        
        else:
            raise ValueError(f"Unknown module type: {module_type}")
    
    @staticmethod
    def visualize(mask: torch.Tensor, title: str = "ASR Attention Mask") -> None:
        """
        Visualize a mask for debugging ASR attention patterns.
        
        Args:
            mask: Mask tensor to visualize
            title: Title for the visualization
        """
        import matplotlib.pyplot as plt
        if mask.dim() == 4:
            # For multi-head attention masks, visualize the first head
            mask_vis = mask[0, 0].cpu().detach().numpy()
        else:
            mask_vis = mask.cpu().detach().numpy()
            
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_vis, cmap='viridis')
        plt.title(title)
        plt.colorbar(label='Attention weight')
        
        # Add grid lines to better visualize the mask structure
        plt.grid(False)
        
        # Add axis labels appropriate for ASR
        if "Audio" in title or "Encoder" in title:
            plt.xlabel("Audio frame position")
            plt.ylabel("Audio frame position")
        elif "Text" in title or "Decoder" in title:
            plt.xlabel("Text token position")
            plt.ylabel("Text token position")
        elif "Cross" in title:
            plt.xlabel("Audio frame position")
            plt.ylabel("Text token position")
            
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()


def create_cross_attention_mask(
    batch_size: int,
    text_seq_len: int,
    audio_seq_len: int,
    num_heads: int,
    audio_padding_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """
    Create a mask for cross-attention from text decoder to audio encoder.
    
    In cross-attention:
    - Each text token should attend to all non-padded audio frames
    - Audio padding mask should have 0s for padded positions and 1s for real data
    
    Args:
        batch_size: Batch size
        text_seq_len: Text sequence length (queries)
        audio_seq_len: Audio sequence length (keys/values)
        num_heads: Number of attention heads
        audio_padding_mask: Mask where 0s indicate audio padding and 1s indicate real data
        device: Device for the tensor
        dtype: Data type for the tensor
        
    Returns:
        torch.Tensor or None: Cross-attention mask or None if no masking needed
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Only create a mask if we need to mask padded audio frames
    if audio_padding_mask is not None:
        # Convert from user format (0=padding, 1=data) to masking format (True=mask, False=keep)
        padding_mask_bool = (audio_padding_mask == 0)
        
        # Expand to [batch_size, 1, 1, audio_seq_len]
        padding_mask_expanded = padding_mask_bool.unsqueeze(1).unsqueeze(2)
        
        # Create cross-attention mask
        # Shape: [batch_size, num_heads, text_seq_len, audio_seq_len]
        cross_attn_mask = torch.zeros(
            (batch_size, num_heads, text_seq_len, audio_seq_len),
            device=device,
            dtype=dtype
        )
        
        # Apply audio padding mask
        # This ensures text tokens don't attend to padded audio frames
        cross_attn_mask = cross_attn_mask.masked_fill(
            padding_mask_expanded.expand(-1, num_heads, text_seq_len, -1),
            float("-inf")
        )
        
        return cross_attn_mask
    
    # No masking needed if audio wasn't padded
    return None

@staticmethod
def get_mask_for_module(
    module_type: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    is_causal: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
    cross_seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """
    Get appropriate mask for specific module types in an ASR model.
    
    Args:
        module_type: Type of module ('audio_encoder', 'text_decoder', 'cross_attention')
        batch_size: Batch size
        seq_len: Primary sequence length
        num_heads: Number of attention heads
        is_causal: Whether causal masking is needed
        padding_mask: Mask where 0s indicate padding, 1s indicate real data
        cross_seq_len: Secondary sequence length (for cross-attention)
        device: Device for the tensor
        dtype: Data type for the tensor
        
    Returns:
        torch.Tensor or None: Appropriate mask for the module or None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if module_type == "audio_encoder":
        # No masking for audio encoder - process all input including padding
        return None
        
    elif module_type == "text_decoder":
        # Always use causal masking for text decoder
        # For text decoder padding, we need to convert from 0=padding to True=padding
        text_padding_mask = None
        if padding_mask is not None:
            text_padding_mask = (padding_mask == 0)
            
        return ASRMask.create_text_decoder_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            padding_mask=text_padding_mask,
            device=device,
            dtype=dtype
        )
        
    elif module_type == "cross_attention":
        if cross_seq_len is None:
            raise ValueError("cross_seq_len must be provided for cross_attention mask")
            
        # For cross-attention, text queries attend to non-padded audio values
        # padding_mask format: 0=padding, 1=data
        return ASRMask.create_cross_attention_mask(
            batch_size=batch_size,
            text_seq_len=seq_len,
            audio_seq_len=cross_seq_len,
            num_heads=num_heads,
            audio_padding_mask=padding_mask,
            device=device,
            dtype=dtype
        )
    
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    
def _canonical_mask(
    mask, mask_name, other_type=None, other_name="", target_type=torch.float32, check_other=True
):
    """Convert mask to canonical form with -inf for masked positions and 0 for unmasked positions.
    
    Args:
        mask: Mask tensor to convert
        mask_name: Name of the mask for error messages
        other_type: Data type of another mask (for compatibility check)
        other_name: Name of the other mask
        target_type: Target data type for the output mask
        check_other: Whether to check compatibility with the other mask
        
    Returns:
        torch.Tensor: Mask in canonical form (-inf for masked positions, 0 for unmasked)
    """
    if mask is None:
        return None
        
    if mask.dtype == torch.bool:
        mask_canonical = torch.zeros_like(mask, dtype=target_type)
        mask_canonical.masked_fill_(mask, float("-inf"))
        return mask_canonical
    
    if check_other and other_type is not None and other_type != mask.dtype:
        raise ValueError(
            f"{mask_name} and {other_name} must have same dtype, but got {mask.dtype} and {other_type}"
        )
        
    return mask.to(target_type)

def create_mask(
    batch_size, 
    head, 
    ctx, 
    is_causal=True, 
    padding_mask=None, 
    device=None, 
    expand_dims=False, 
    mask_type="bool", 
    register_buffer=False, 
    module=None,
    key_padding_mask=None,
    attn_mask=None,
    is_encoder=False,
    is_decoder=True,
    encoder_decoder_cross=False,
    query_dtype=None,
    head_mask=None,
    layer_mask=None
):
    """Create attention masks for transformer models.
    
    Args:
        batch_size: Batch size
        head: Number of attention heads
        ctx: Sequence length
        is_causal: Whether to apply causal masking
        padding_mask: Mask for padding tokens (True means masked)
        device: Device for the mask tensor
        expand_dims: Whether to expand dimensions for batch and heads
        mask_type: Type of mask ('bool', 'float', 'neg_inf', 'canonical')
        register_buffer: Whether to register the mask as a buffer in a module
        module: Module to register the buffer in
        key_padding_mask: Mask for padding in the key sequence
        attn_mask: Additional attention mask to combine with other masks
        is_encoder: Whether this mask is for an encoder
        is_decoder: Whether this mask is for a decoder
        encoder_decoder_cross: Whether this mask is for cross-attention
        query_dtype: Data type for the query tensor
        head_mask: Mask for specific heads
        layer_mask: Mask for specific layers
        
    Returns:
        torch.Tensor: The created mask
    """
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Handle encoder-only case
    if is_encoder and not is_decoder:
        if mask_type == "bool":
            mask = torch.zeros((ctx, ctx), device=device).bool()
        elif mask_type == "float":
            mask = torch.zeros((ctx, ctx), device=device).float()
        elif mask_type == "neg_inf":
            mask = torch.zeros((ctx, ctx), device=device)
        elif mask_type == "canonical":
            mask = torch.zeros((ctx, ctx), device=device, dtype=query_dtype or torch.float32)
            
        if padding_mask is not None:
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            if mask_type == "bool":
                mask = mask | padding_mask_expanded
            else:
                # Convert padding_mask to float and apply
                mask = mask + padding_mask_expanded.float() * float("-inf")
    
    # Handle decoder-only case        
    elif is_decoder and not encoder_decoder_cross:
        if is_causal:
            if mask_type == "bool":
                mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).bool()
            elif mask_type == "float":
                mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).float()
            elif mask_type == "neg_inf":
                mask = torch.triu(torch.full((ctx, ctx), float("-inf"), device=device), diagonal=1)
            elif mask_type == "canonical":
                mask = torch.triu(torch.full((ctx, ctx), float("-inf"), device=device), diagonal=1)
                
            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
                if mask_type == "bool":
                    mask = mask | key_padding_mask_expanded
                else:
                    # Combine causal mask with padding mask
                    key_padding_float = key_padding_mask_expanded.float() * float("-inf")
                    mask = mask + key_padding_float
        else:
            # Non-causal decoder mask (only padding mask if available)
            if mask_type == "bool":
                mask = torch.zeros((ctx, ctx), device=device).bool()
            elif mask_type == "float":
                mask = torch.zeros((ctx, ctx), device=device).float()
            elif mask_type == "neg_inf":
                mask = torch.zeros((ctx, ctx), device=device)
            elif mask_type == "canonical":
                mask = torch.zeros((ctx, ctx), device=device, dtype=query_dtype or torch.float32)
                
            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
                if mask_type == "bool":
                    mask = mask | key_padding_mask_expanded
                else:
                    # Apply padding mask
                    key_padding_float = key_padding_mask_expanded.float() * float("-inf")
                    mask = mask + key_padding_float
    
    # Handle cross-attention case        
    elif encoder_decoder_cross:
        if mask_type == "bool":
            mask = torch.zeros((ctx, ctx), device=device).bool()
        elif mask_type == "float":
            mask = torch.zeros((ctx, ctx), device=device).float()
        elif mask_type == "neg_inf":
            mask = torch.zeros((ctx, ctx), device=device)
        elif mask_type == "canonical":
            mask = torch.zeros((ctx, ctx), device=device, dtype=query_dtype or torch.float32)
            
        # For cross-attention, only key_padding_mask is typically used
        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if mask_type == "bool":
                mask = mask | key_padding_mask_expanded
            else:
                key_padding_float = key_padding_mask_expanded.float() * float("-inf")
                mask = mask + key_padding_float
    
    # Default case
    else:
        if mask_type == "bool":
            mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).bool() if is_causal else torch.zeros((ctx, ctx), device=device).bool()
        elif mask_type == "float":
            mask = torch.triu(torch.ones((ctx, ctx), device=device), diagonal=1).float() if is_causal else torch.zeros((ctx, ctx), device=device).float()
        elif mask_type == "neg_inf":
            mask = torch.zeros((ctx, ctx), device=device)
            if is_causal:
                mask = mask.triu_(1).fill_(float("-inf"))
        elif mask_type == "canonical":
            mask = torch.zeros((ctx, ctx), device=device, dtype=query_dtype or torch.float32)
            if is_causal:
                mask = mask.triu_(1).fill_(float("-inf"))
    
    # Register as buffer if requested
    if register_buffer:
        if module is None:
            raise ValueError("Module must be provided when register_buffer=True")
        module.register_buffer("mask", mask, persistent=False)
    
    # Expand dimensions for batch and heads if requested
    if expand_dims:
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, head, ctx, ctx)
        
        # Apply head mask if provided
        if head_mask is not None:
            head_mask = _prepare_head_mask(head_mask, head, mask_type, device)
            
            head_mask_expanded = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if mask_type == "bool":
                mask = mask | head_mask_expanded.expand(batch_size, -1, ctx, ctx)
            elif mask_type in ["neg_inf", "canonical"]:
                head_mask_inf = head_mask_expanded * float("-inf")
                mask = mask + head_mask_inf.expand(batch_size, -1, ctx, ctx)
        
        # For layer masks, we don't apply them to the attention mask directly
        if layer_mask is not None and module is not None:
            print("Layer masking will be applied at the model level, not in the attention mask.")
    
    return mask

def _prepare_head_mask(head_mask, num_heads, mask_type, device):
    """Prepares a head mask for use in attention masks"""
    if head_mask.dim() == 1:
        if head_mask.size(0) != num_heads:
            raise ValueError(f"Head mask size {head_mask.size(0)} doesn't match number of heads {num_heads}")
        return head_mask.to(device)
    else:
        raise ValueError(f"Head mask should be 1D, but got {head_mask.dim()}D")

def combine_masks(mask1, mask2, mask_type="bool"):
    """Combine two masks based on mask_type"""
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
        
    if mask_type == "bool":
        return mask1 | mask2
    else:  # float, neg_inf, canonical
        return mask1 + mask2

# Helper function to create a BYOM (Bring Your Own Mask) attention mask
def create_byom_mask(mask_tensor, batch_size, num_heads, seq_len, mask_type="neg_inf"):
    """
    Convert a custom mask tensor into the appropriate format for attention
    
    Args:
        mask_tensor: Custom mask tensor (True means masked positions)
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        mask_type: Type of mask ('bool', 'float', 'neg_inf')
        
    Returns:
        torch.Tensor: Formatted attention mask
    """
    if mask_tensor.dim() == 2:  # [batch_size, seq_len]
        mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        mask_tensor = mask_tensor.expand(batch_size, num_heads, seq_len, seq_len)
    elif mask_tensor.dim() == 3:  # [batch_size, seq_len, seq_len]
        mask_tensor = mask_tensor.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        mask_tensor = mask_tensor.expand(batch_size, num_heads, seq_len, seq_len)
    elif mask_tensor.dim() != 4:  # Not [batch_size, num_heads, seq_len, seq_len]
        raise ValueError(f"Mask tensor has unsupported shape: {mask_tensor.shape}")
    
    if mask_type == "bool":
        return mask_tensor.bool()
    elif mask_type == "float":
        return mask_tensor.float()
    elif mask_type == "neg_inf":
        result = torch.zeros_like(mask_tensor, dtype=torch.float)
        result.masked_fill_(mask_tensor, float("-inf"))
        return result
    else:
        raise ValueError(f"Unsupported mask_type: {mask_type}")

def create_masks(encoder_input, decoder_input, pad_token_id=0):
    batch_size, encoder_seq_len = encoder_input.size()
    _, decoder_seq_len = decoder_input.size()
    # Decoder mask: Causal mask for self-attention
    decoder_causal_mask = torch.triu(torch.ones((decoder_seq_len, decoder_seq_len), dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)  # (1, 1, decoder_seq_len, decoder_seq_len)

    # Cross-attention mask: Only encoder padding mask
    encoder_padding_mask = encoder_input.eq(pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, encoder_seq_len)
    return decoder_causal_mask, encoder_padding_mask

def visualize_mask(mask, title="Attention Mask"):
    import matplotlib.pyplot as plt
    if mask.dim() == 4:
        mask_vis = mask[0, 0].cpu().detach().numpy()
    else:
        mask_vis = mask.cpu().detach().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(mask_vis, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

