"""
Qwen API Client - Sample implementation for interacting with Qwen API
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)
from dataclasses import dataclass

from core.config import LLM_CONFIG, API_KEY_ENV_VARS, _normalize_chat_completions_url


@dataclass
class QwenMessage:
    """Represents a message in the conversation"""
    role: str  # "system", "user", or "assistant"
    content: str


class QwenAPIClient:
    """Client for interacting with Qwen API"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Qwen API client
        
        Args:
            api_key: Your Qwen API key (can also be set via QWEN_API_KEY env var)
            base_url: Base URL for Qwen API (optional, accepts either an API root or full endpoint)
        """
        self.api_key = api_key or os.getenv(API_KEY_ENV_VARS["qwen"])
        resolved_base_url = base_url or os.getenv("QWEN_BASE_URL") or LLM_CONFIG["qwen"]["base_url"]
        self.base_url = _normalize_chat_completions_url(resolved_base_url)
        self.legacy_base_url = LLM_CONFIG["qwen"]["legacy_base_url"]
        self.legacy_models = LLM_CONFIG["qwen"]["legacy_models"]
        self.default_model = os.getenv("QWEN_MODEL") or LLM_CONFIG["qwen"]["default_model"]
        
        if not self.api_key:
            raise ValueError(f"API key is required. Set {API_KEY_ENV_VARS['qwen']} environment variable or pass api_key parameter.")
    
    def _is_legacy_model(self, model: str) -> bool:
        """Check if model uses legacy endpoint"""
        return model in self.legacy_models
    
    def _make_request(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Make HTTP request to Qwen API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use legacy endpoint for old models
        url = self.legacy_base_url if self._is_legacy_model(model) else self.base_url
        
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=240)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as e:
                if attempt < max_attempts:
                    logger.warning(f"API request timed out (attempt {attempt}/{max_attempts}), retrying...")
                    continue
                raise Exception(f"API request failed: {e}")
            except requests.exceptions.HTTPError as e:
                if response.status_code >= 500 and attempt < max_attempts:
                    wait = 3 * attempt
                    logger.warning(f"Server error ({response.status_code}), waiting {wait}s before retry {attempt}/{max_attempts}...")
                    time.sleep(wait)
                    continue
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise Exception(f"API request failed: {e}\nResponse: {error_detail}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {e}")
    
    def chat_completion(
        self,
        messages: List[QwenMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Qwen API
        
        Args:
            messages: List of conversation messages
            model: Model to use (qwen-turbo, qwen-plus, qwen-max, qwen3.5-flash, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            
        Returns:
            API response dictionary
        """
        # Use default values from config if not provided
        model = model or self.default_model
        max_tokens = max_tokens or LLM_CONFIG["qwen"]["default_params"]["max_tokens"]
        temperature = temperature if temperature is not None else LLM_CONFIG["qwen"]["default_params"]["temperature"]
        top_p = top_p or LLM_CONFIG["qwen"]["default_params"]["top_p"]
        stream = stream if stream is not None else LLM_CONFIG["qwen"]["default_params"]["stream"]
        
        # Format payload based on model type
        if self._is_legacy_model(model):
            # Legacy DashScope format
            payload = {
                "model": model,
                "input": {
                    "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
                },
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "incremental_output": stream
                }
            }
        else:
            # OpenAI-compatible format for new models
            payload = {
                "model": model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream
            }
        
        return self._make_request(payload, model)
    
    def simple_chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Simple chat interface - send a prompt and get response
        
        Args:
            prompt: User prompt/question
            model: Model to use (optional, uses config value if not provided)
            
        Returns:
            Generated response text
        """
        # Use default model from config if not provided
        model = model or self.default_model
        
        messages = [QwenMessage(role="user", content=prompt)]
        response = self.chat_completion(messages, model=model, temperature=temperature)
        
        try:
            # Try OpenAI-compatible format first (new models)
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            # Fall back to legacy format (old models)
            elif "output" in response:
                return response["output"]["text"]
            else:
                raise Exception(f"Unexpected response format: {response}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse response: {e}\nResponse: {response}")
    
    def conversation_chat(
        self,
        messages: List[QwenMessage],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Multi-turn conversation chat
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt to set context
            model: Model to use (optional, uses config value if not provided)
            
        Returns:
            Generated response text
        """
        # Use default model from config if not provided
        model = model or self.default_model
        
        conversation = []
        
        if system_prompt:
            conversation.append(QwenMessage(role="system", content=system_prompt))
        
        conversation.extend(messages)
        
        response = self.chat_completion(conversation, model=model)
        
        try:
            # Try OpenAI-compatible format first (new models)
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            # Fall back to legacy format (old models)
            elif "output" in response:
                return response["output"]["text"]
            else:
                raise Exception(f"Unexpected response format: {response}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse response: {e}\nResponse: {response}")


def main():
    """Example usage of Qwen API client"""
    
    # Initialize client (make sure to set QWEN_API_KEY environment variable)
    try:
        client = QwenAPIClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your QWEN_API_KEY environment variable")
        return
    
    print("=== Qwen API Client Demo ===\n")
    
    # Example 1: Simple chat
    print("1. Simple Chat Example:")
    try:
        response = client.simple_chat("Hello! Can you tell me about artificial intelligence?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Conversation with system prompt
    print("2. Conversation with System Prompt:")
    try:
        system_prompt = "You are a helpful assistant that specializes in explaining technical concepts clearly."
        messages = [
            QwenMessage(role="user", content="What is machine learning?"),
        ]
        
        response = client.conversation_chat(messages, system_prompt=system_prompt)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Multi-turn conversation
    print("3. Multi-turn Conversation:")
    try:
        messages = [
            QwenMessage(role="user", content="What is Python?"),
            QwenMessage(role="assistant", content="Python is a high-level programming language known for its simplicity and readability."),
            QwenMessage(role="user", content="Can you give me a simple Python example?")
        ]
        
        response = client.conversation_chat(messages)
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 4: Different models and parameters
    print("4. Using Different Model and Parameters:")
    try:
        messages = [QwenMessage(role="user", content="Write a short poem about technology")]
        response = client.chat_completion(
            messages,
            model="qwen-plus",  # Using a different model
            temperature=0.9,    # Higher creativity
            max_tokens=200
        )
        print(f"Response: {response['output']['text']}\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
