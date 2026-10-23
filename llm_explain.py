"""
LLM-powered explanation generator for retinal disease predictions.

Uses Google Gemini (primary) or Groq (fallback) to produce a human-readable
explanation of *why* the model flagged a particular disease, based on:
  - the disease code and name,
  - the model's confidence (probability),
  - the known clinical signs for that condition,
  - the full set of class probabilities.

Set your API keys as environment variables:
    GEMINI_API_KEY   – for Google Gemini
    GROQ_API_KEY     – for Groq (fallback)

If neither key is set, a static fallback explanation is returned.
"""

import os
import json
import traceback

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from disease_info import INFO, F1

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a medical AI assistant specialising in ophthalmology.  A deep-learning
model has just analysed a retinal fundus photograph and flagged one or more
eye conditions.  Your job is to explain, in clear everyday language, WHY the
model likely made this prediction.

Rules
─────
1. Base your explanation on the clinical signs the model was trained to
   recognise (provided below) and how they relate to the condition.
2. DO NOT diagnose the patient – you are explaining the model's reasoning,
   not issuing a medical opinion.
3. Mention the model's confidence level and what it means.
4. If the model's F1 score for this condition is below 0.5, note that the
   model is weak on this particular condition.
5. Keep the tone professional yet accessible – a non-specialist should
   understand you.
6. Use **markdown** formatting: bold key terms, use bullet points for signs.
7. End with a short disclaimer that this is an AI explanation, not a diagnosis.
8. Keep the response between 150-300 words.
"""


def _build_user_prompt(disease_code, prob, all_probs, single_eye=False):
    """Build the user-facing prompt sent to the LLM."""
    info = INFO.get(disease_code, {})
    name = info.get("name", disease_code)
    what = info.get("what", "")
    signs = info.get("signs", "")
    risk = info.get("risk", "")
    f1 = F1.get(disease_code, 0)

    other_scores = "\n".join(
        f"  • {INFO.get(c, {}).get('name', c)}: {p:.1%}"
        for c, p in sorted(all_probs.items(), key=lambda t: -t[1])
        if c != disease_code
    )

    return f"""\
The model flagged **{name}** (code: {disease_code}) with a confidence of
{prob:.1%}.

About this condition:
  {what}

Clinical signs the model looks for:
  {signs}

Why it matters:
  {risk}

Model reliability (F1 on validation data): {f1:.2f}
{"⚠️ This is below 0.5, so the model is weak on this condition." if f1 < 0.5 else ""}

Other class scores for context:
{other_scores}

{"Note: only one eye image was provided; the model used it for both eyes, which may reduce accuracy." if single_eye else ""}

Please explain, in plain language, why the model likely predicted this
condition based on the signs it was trained to detect.  Relate the clinical
signs to how they would appear in a fundus photograph.
"""


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def _ask_gemini(system, user):
    """Call the Google Gemini API."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system,
    )
    response = model.generate_content(user, request_options={"timeout": 5.0})
    return response.text


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
def _ask_groq(system, user):
    """Call the Groq API with retry and timeout."""
    import time
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    client = Groq(api_key=api_key, timeout=30.0)

    for attempt in range(3):
        try:
            chat = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            return chat.choices[0].message.content
        except Exception:
            if attempt < 2:
                time.sleep(1.5)
            else:
                raise
    return None


# ---------------------------------------------------------------------------
# Static fallback (no API key available)
# ---------------------------------------------------------------------------
def _static_fallback(disease_code, prob):
    info = INFO.get(disease_code, {})
    name = info.get("name", disease_code)
    signs = info.get("signs", "No information available.")
    what = info.get("what", "")
    risk = info.get("risk", "")
    f1 = F1.get(disease_code, 0)

    return f"""\
**Why the model flagged {name}**

The model detected patterns in the retinal fundus image that are consistent
with **{name}** (confidence: {prob:.1%}).

**What it looked for:**
{signs}

**About this condition:**
{what}

**Why it matters:**
{risk}

**Model reliability:** F1 = {f1:.2f} on its own validation data.
{"⚠️ The model is weak on this condition (F1 < 0.5)." if f1 < 0.5 else ""}

> *This is an automated explanation based on the model's training data,
> not a clinical diagnosis. Please consult an eye-care professional.*
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def explain_prediction(disease_code, prob, all_probs, single_eye=False):
    """
    Return a dict with keys:
        explanation (str) – markdown-formatted explanation
        provider    (str) – 'gemini', 'groq', or 'static'
        error       (str | None)
    """
    user_prompt = _build_user_prompt(disease_code, prob, all_probs, single_eye)

    # Try Gemini first
    try:
        text = _ask_gemini(_SYSTEM_PROMPT, user_prompt)
        if text:
            return {"explanation": text, "provider": "gemini", "error": None}
    except Exception:
        traceback.print_exc()

    # Fall back to Groq
    try:
        text = _ask_groq(_SYSTEM_PROMPT, user_prompt)
        if text:
            return {"explanation": text, "provider": "groq", "error": None}
    except Exception:
        traceback.print_exc()

    # Static fallback
    return {
        "explanation": _static_fallback(disease_code, prob),
        "provider": "static",
        "error": "No LLM API key found. Set GEMINI_API_KEY or GROQ_API_KEY for AI-powered explanations.",
    }
