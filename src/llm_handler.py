import requests

DEFAULT_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

class LLMHandler:
    def __init__(self, api_token, model_url=None):
        self.api_token = api_token
        self.model_url = model_url or DEFAULT_MODEL_URL

    def _call_api(self, prompt, max_new_tokens=256):
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
            },
        }
        try:
            response = requests.post(self.model_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Hugging Face API request timed out. The model may be loading — please try again.")
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Could not connect to Hugging Face API: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Hugging Face API returned an error ({exc.response.status_code}). "
                "Check that your HF_API_TOKEN is valid and the model is accessible."
            ) from exc
        result = response.json()
        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text", "")
            if "[/INST]" in generated_text:
                return generated_text.split("[/INST]")[-1].strip()
            return generated_text.strip()
        return ""

    def generate_response(self, prompt):
        llama_prompt = f"[INST] {prompt} [/INST]"
        return self._call_api(llama_prompt)

    def summarize_document(self, document):
        prompt = (
            f"[INST] You are a helpful medical assistant. "
            f"Please provide a concise summary of the following document:\n\n{document} [/INST]"
        )
        return self._call_api(prompt)

    def answer_medical_question(self, question, context=""):
        if context:
            prompt = (
                f"[INST] You are a helpful medical report analyzer. "
                f"Based on the following context, answer the question clearly and accurately.\n\n"
                f"Context:\n{context}\n\nQuestion: {question} [/INST]"
            )
        else:
            prompt = f"[INST] You are a helpful medical assistant. {question} [/INST]"
        return self._call_api(prompt)