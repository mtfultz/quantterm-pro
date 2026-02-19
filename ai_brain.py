"""
AI Brain - The Intelligence Layer
Connects to local Ollama (mixtral) to make trading decisions based on context.
"""

import json
import logging
from ollama import Client as OllamaClient
from datetime import datetime
from typing import Dict, Optional
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIBrain:
    """
    AI-powered decision maker using local LLM (Mixtral via Ollama).
    Analyzes market context and returns structured trading signals.
    """

    def __init__(self, model: str = None, timeout: int = None):
        """
        Initialize the AI Brain.

        Args:
            model: Ollama model name (default: from config)
            timeout: Request timeout in seconds (default: from config)
        """
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.request_count = 0
        self._client = OllamaClient(host=config.OLLAMA_BASE_URL)

        logger.info(f"AIBrain initialized with model: {self.model}, host: {config.OLLAMA_BASE_URL}")
        self._verify_ollama_connection()

    def _verify_ollama_connection(self) -> bool:
        """Verify that Ollama is running and the model is available."""
        try:
            # List available models on the configured host
            models = self._client.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Available: {model_names}")
                logger.warning(f"Run: ollama pull {self.model}")
                return False

            logger.info(f"Ollama connection verified. Model '{self.model}' is ready.")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: sudo systemctl start ollama")
            return False

    def _construct_system_prompt(self) -> str:
        """
        Construct the system prompt that defines the AI's role and output format.
        """
        return """You are an elite hedge fund trader with 20 years of experience in technical analysis and market psychology.

Your job: Analyze the provided market data and decide whether to BUY or HOLD.

CRITICAL RULES:
1. You MUST respond ONLY with valid JSON. No explanations, no markdown, just JSON.
2. Use this EXACT format:
{
  "signal": "BUY" or "HOLD",
  "confidence": 0.85,
  "reasoning": "Brief explanation (max 100 chars)"
}

3. "confidence" must be a float between 0.0 (no confidence) and 1.0 (maximum confidence)
4. "signal" must be exactly "BUY" or "HOLD" (uppercase)
5. "reasoning" should be concise and actionable

DECISION CRITERIA:
- BUY: Strong bullish setup with clear risk/reward
- HOLD: Uncertain conditions, wait for better setup

Consider:
- Price action relative to moving averages
- RSI momentum (oversold = potential bounce)
- Recent volatility and trend strength
- Risk/reward ratio for the setup"""

    def _construct_user_prompt(self, context_text: str) -> str:
        """
        Format the market context into a user prompt.

        Args:
            context_text: Market data and indicators as a string

        Returns:
            Formatted user prompt
        """
        return f"""MARKET ANALYSIS REQUEST

{context_text}

Based on this data, should I BUY or HOLD? Respond with JSON only."""

    def analyze_market(self, context_text: str, retry_count: int = 2) -> Dict:
        """
        Analyze market context using the AI model and return structured decision.

        Args:
            context_text: String containing market data and indicators
            retry_count: Number of retries on failure

        Returns:
            Dict with keys: 'signal', 'confidence', 'reasoning'
            On error, returns safe default: {'signal': 'HOLD', 'confidence': 0.0, ...}
        """
        self.request_count += 1
        request_id = f"REQ_{self.request_count}_{datetime.now().strftime('%H%M%S')}"

        logger.info(f"[{request_id}] Sending analysis request to AI...")
        logger.debug(f"[{request_id}] Context: {context_text[:200]}...")

        for attempt in range(retry_count + 1):
            try:
                # Call Ollama on the configured remote/local host
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': self._construct_system_prompt()
                        },
                        {
                            'role': 'user',
                            'content': self._construct_user_prompt(context_text)
                        }
                    ],
                    format='json',  # Forces JSON output
                    options={
                        'temperature': 0.55,  # Lower = more consistent
                        'top_p': 0.9,
                        'num_predict': 150,  # Limit response length
                    }
                )

                # Extract the response content
                raw_response = response['message']['content']
                logger.info(f"[{request_id}] Raw AI response: {raw_response}")

                # Parse JSON
                result = json.loads(raw_response)

                # Validate structure
                required_keys = ['signal', 'confidence', 'reasoning']
                if not all(key in result for key in required_keys):
                    raise ValueError(f"Missing required keys. Got: {result.keys()}")

                # Validate signal
                if result['signal'] not in ['BUY', 'HOLD']:
                    logger.warning(f"Invalid signal '{result['signal']}', defaulting to HOLD")
                    result['signal'] = 'HOLD'

                # Validate confidence
                result['confidence'] = float(result['confidence'])
                if not 0.0 <= result['confidence'] <= 1.0:
                    logger.warning(f"Invalid confidence {result['confidence']}, clamping to [0,1]")
                    result['confidence'] = max(0.0, min(1.0, result['confidence']))

                # Log successful parse
                logger.info(
                    f"[{request_id}] ✓ Decision: {result['signal']} "
                    f"(confidence: {result['confidence']:.2f}) - {result['reasoning']}"
                )

                return result

            except json.JSONDecodeError as e:
                logger.error(f"[{request_id}] Attempt {attempt+1}/{retry_count+1} - JSON parse error: {e}")
                logger.error(f"[{request_id}] Raw response was: {raw_response if 'raw_response' in locals() else 'N/A'}")

            except ollama.ResponseError as e:
                logger.error(f"[{request_id}] Attempt {attempt+1}/{retry_count+1} - Ollama error: {e}")

            except Exception as e:
                logger.error(f"[{request_id}] Attempt {attempt+1}/{retry_count+1} - Unexpected error: {e}")

            # Wait before retry (except on last attempt)
            if attempt < retry_count:
                import time
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"[{request_id}] Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed - return safe default
        logger.error(f"[{request_id}] All retry attempts failed. Returning HOLD signal.")
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'reasoning': 'AI analysis failed - defaulting to safe HOLD'
        }

    def get_stats(self) -> Dict:
        """Return statistics about AI usage."""
        return {
            'model': self.model,
            'total_requests': self.request_count,
            'timeout': self.timeout
        }


# Standalone test
if __name__ == "__main__":
    print("Testing AIBrain...")
    print("=" * 60)

    # Initialize
    brain = AIBrain()

    # Test context
    test_context = """
    Symbol: SPY
    Current Price: $550.25
    SMA 200: $540.00
    RSI 14: 38.5

    Analysis:
    - Price is 1.9% above 200-day moving average (uptrend confirmed)
    - RSI is in oversold territory (38.5 < 45), indicating potential bounce
    - Recent pullback of -2.3% from recent high
    - Volume is above average, showing strong participation
    """

    # Get decision
    decision = brain.analyze_market(test_context)

    print("\n" + "=" * 60)
    print("AI DECISION:")
    print(f"  Signal:     {decision['signal']}")
    print(f"  Confidence: {decision['confidence']:.1%}")
    print(f"  Reasoning:  {decision['reasoning']}")
    print("=" * 60)
    print(f"\nStats: {brain.get_stats()}")
