# backend/agents/classifier_agent.py

from ..groq_client import chat_completion_text

# All allowed intent labels your chat router expects
INTENT_LABELS = [
    "missing_info",
    "get_sebi_rule",
    "build_portfolio",
    "run_simulation",
    "final_advice",
    "chat_general",
]


class ClassifierAgent:
    """
    Simple intent classifier using Azure GPT.

    It must always return exactly ONE of:
      - missing_info
      - get_sebi_rule
      - build_portfolio
      - run_simulation
      - final_advice
      - chat_general
    """

    SYSTEM_PROMPT = """
You are an intent classifier for a financial advisory assistant.

Given a single user message, decide which ONE of the following intent labels best matches it:

1. missing_info
   - The user is asking for advice that clearly requires more profile data
     (age, income, tenure, SIP amount, risk profile) which is not yet present.
   - Example: "Make a plan for me", "Suggest investments" when key data is missing.

2. get_sebi_rule
   - The user is asking about regulations, SEBI rules, mutual fund categories, tax sections,
     or any factual legal/compliance detail.
   - Example: "What are SEBI rules for liquid funds?", "Explain ELSS tax benefit".

3. build_portfolio
   - The user is explicitly asking to construct or change a portfolio or allocation.
   - Example: "Build my portfolio", "Suggest allocation", "How should I split equity and debt?".

4. run_simulation
   - The user is asking to project or simulate future returns, probabilities, or outcomes.
   - Example: "Simulate my SIP", "What will my corpus be in 20 years?", "Run a Monte Carlo".

5. final_advice
   - The user is asking for a consolidated final recommendation or full financial plan.
   - Example: "Give me my full financial plan", "Summarize everything and advise me".

6. chat_general
   - General conversation, greetings, small questions, definitions, or anything else
     that does not require RAG, portfolio construction, simulation, or missing info collection.
   - Example: "Hi", "What is SIP?", "Explain mutual funds simply".

Return ONLY the intent label string. Do not add quotes, explanations, or extra text.
"""

    def classify(self, user_message: str) -> str:
        """
        Classify the user's message into one of INTENT_LABELS.
        Falls back to 'chat_general' if anything unexpected happens.
        """
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            # Rule-based check BEFORE GPT classification
            # If message indicates user wants a portfolio or simulation, but required data missing → missing_info
            lower_msg = user_message.lower()

            trigger_phrases = ["build my portfolio", "create portfolio", "suggest allocation",
                            "simulate", "run simulation", "expected value", "best case", "worst case"]

            if any(tp in lower_msg for tp in trigger_phrases):
                # Check memory here
                from ..memory.store import memory_store
                entity = memory_store.get_entity(session_id=None)  # temporary, will inject session_id later if needed

                needed_fields = ["age", "risk_category", "tenure_years"]
                if not all(f in entity for f in needed_fields):
                    return "missing_info"

            raw = chat_completion_text(messages, temperature=0.0)
            label = (raw or "").strip().lower()

            if label not in INTENT_LABELS:
                return "chat_general"

            return label

        except Exception as ex:
            # In case of any failure, do not break chat
            print(f"[ClassifierAgent] Error during classification: {ex}")
            return "chat_general"


classifier_agent = ClassifierAgent()


# Simple manual test when running this file directly
if __name__ == "__main__":
    tests = [
        "Build my portfolio",
        "Run a simulation for my SIP",
        "What are SEBI rules for liquid funds?",
        "Give me my complete financial plan",
        "Hi",
        "Suggest investments for me",
    ]
    for t in tests:
        print(f"{t!r} -> {classifier_agent.classify(t)}")
