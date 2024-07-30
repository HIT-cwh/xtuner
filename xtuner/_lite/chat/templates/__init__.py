from .chat import ChatTemplate
from .hybrid import HybridChatTemplate

CHAT_TEMPLATE_MAP = {
    'internlm2':
    HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>']),
    'deepseek_v2':
    HybridChatTemplate(
        system='{system}\n\n',
        user='User: {user}\n\nAssistant: ',
        assistant='{assistant}<｜end▁of▁sentence｜>',
        stop_words=['<｜end▁of▁sentence｜>']),
}

__all__ = ['ChatTemplate', 'HybridChatTemplate']
