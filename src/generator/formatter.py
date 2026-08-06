class ContextFormatter:
    def format_prompt(self, query: str, context: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are a helpful, concise AI assistant. Answer the user's question directly using ONLY the provided context.\n"
            "Keep your response brief, summarizing the answer in exactly 2 to 3 sentences.\n"
            "If the answer is not present in the context, state clearly that you do not know.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Context: {context}\n\nQuestion: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )