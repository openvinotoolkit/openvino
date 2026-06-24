"""HuggingFace model loader with caching and validation."""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import json

from ..utils import get_logger


class HuggingFaceLoader:
    """Loads and manages HuggingFace models with caching."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the loader.
        
        Args:
            cache_dir: Directory for caching models
        """
        self.logger = get_logger(self.__class__.__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model_and_tokenizer(
        self,
        model_name: str,
        torch_dtype: str = "float32",
        trust_remote_code: bool = True,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, AutoConfig]:
        """Load model, tokenizer, and config from HuggingFace.
        
        Args:
            model_name: HuggingFace model identifier
            torch_dtype: PyTorch data type (float32, float16, bfloat16)
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading
            
        Returns:
            Tuple of (model, tokenizer, config)
        """
        self.logger.info(f"Loading model: {model_name}")
        
        # Convert string dtype to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto"
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.float32)
        
        try:
            # Load config first to validate model
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                cache_dir=self.cache_dir
            )
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                cache_dir=self.cache_dir
            )
            
            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.info("Set pad_token to eos_token")
            
            # Load model
            self.logger.info("Loading model...")
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "cache_dir": self.cache_dir,
                "torch_dtype": torch_dtype_obj,
                **kwargs
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Set model to evaluation mode
            model.eval()
            
            self.logger.info(f"Successfully loaded {model_name}")
            self.logger.info(f"Model config: {config}")
            self.logger.info(f"Model parameters: {self._count_parameters(model):,}")
            
            return model, tokenizer, config
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def validate_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        test_input: str = "Hello, this is a test."
    ) -> bool:
        """Validate that model and tokenizer work correctly.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            test_input: Test string for validation
            
        Returns:
            True if validation passes
        """
        try:
            self.logger.info("Validating model and tokenizer...")
            
            # Tokenize input
            inputs = tokenizer(
                test_input,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Check output shape
            logits = outputs.logits
            expected_shape = (inputs.input_ids.shape[0], inputs.input_ids.shape[1], model.config.vocab_size)
            
            if logits.shape != expected_shape:
                self.logger.error(f"Unexpected output shape: {logits.shape}, expected: {expected_shape}")
                return False
            
            self.logger.info("Model validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def get_model_info(self, model: PreTrainedModel, config: AutoConfig) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Args:
            model: Loaded model
            config: Model configuration
            
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": getattr(config, "hidden_size", "N/A"),
            "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", "N/A")),
            "num_attention_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", "N/A")),
            "max_position_embeddings": getattr(config, "max_position_embeddings", getattr(config, "n_positions", "N/A")),
            "total_parameters": self._count_parameters(model),
            "trainable_parameters": self._count_parameters(model, trainable_only=True),
            "model_size_mb": self._estimate_model_size(model)
        }
        
        return info
    
    def _count_parameters(self, model: PreTrainedModel, trainable_only: bool = False) -> int:
        """Count model parameters.
        
        Args:
            model: PyTorch model
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    def _estimate_model_size(self, model: PreTrainedModel) -> float:
        """Estimate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Estimated size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return round(size_mb, 2)