from .deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM
from .internlm2 import InternLM2Config, InternLM2ForCausalLM
from .llava.configuration_llava import EnhancedLlavaConfig
from .llava.modeling_llava import LlavaForConditionalGeneration
from .llava.processing_llava import LlavaProcessor


def register_remote_code():
    from transformers import AutoConfig, AutoModelForCausalLM
    AutoConfig.register('internlm2', InternLM2Config, exist_ok=True)
    AutoModelForCausalLM.register(
        InternLM2Config, InternLM2ForCausalLM, exist_ok=True)
