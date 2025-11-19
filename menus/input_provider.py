from typing import Optional, Dict

class InputProvider:
    """Provides answers to menu prompts."""
    
    def __init__(self, preloaded: Optional[Dict[str, str]] = None):
        self.preloaded = preloaded or {}
    
    def get(self, prompt_id: str, default: Optional[str] = None) -> Optional[str]:
        """Return the preloaded value for this prompt, or None."""
        return self.preloaded.get(prompt_id, default)
