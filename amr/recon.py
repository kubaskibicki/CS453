import json
import os
from .interfaces import FunctionProfile, RunConfig

CACHE_DIR = ".llm_cache"

PROMPT_TEMPLATE = """You are analyzing the real-valued mathematical function {name}(x).
Respond with ONLY a JSON object, no prose, with exactly these keys:
- "symmetry": one of "even", "odd", "none"
- "periodic": true or false
- "period": the period as a number, or null if not periodic
- "monotonic": one of "increasing", "decreasing", "none"
- "domain": [low, high] over which the function is defined and well-behaved
- "range": [low, high] of output values
Function to analyze: {name}
"""


def _chat_json(model, prompt):
    """Send a prompt to Ollama model and return the json response
    
    Args:
        model (str): Name of the Ollama model to use
        prompt (str): User prompt sent to the model
    
    Returns:
        dict: Parsed json object returned by the model
    """
    import ollama  # we dont want tests to need to have ollama installed
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(resp["message"]["content"])


def _parse_profile(name, raw):
    """Convert raw profile data into a FunctionProfile object

    Missing or invalid domain/range values are replaced with default
    bounds.

    Args:
        name (str): Name of the function
        raw (dict): Raw profile data returned by the LLM

    Returns:
        FunctionProfile: Parsed function profile
    """
    domain = raw.get("domain")
    if not domain or len(domain) < 2 or domain[0] is None or domain[1] is None:
        domain = [-100.0, 100.0]
    rng = raw.get("range")
    if not rng or len(rng) < 2 or rng[0] is None or rng[1] is None:
        rng = [-100.0, 100.0]
    return FunctionProfile(
        name=name,
        symmetry=raw.get("symmetry", "none") or "none",
        periodic=bool(raw.get("periodic", False)),
        period=raw.get("period", None),
        monotonic=raw.get("monotonic", "none") or "none",
        domain=(float(domain[0]), float(domain[1])),
        output_range=(float(rng[0]), float(rng[1])),
    )


def _cache_path(model, name):
    """Generate the cache file path for a function and model

    Args:
        model (str): Name of the AI model
        name (str): Name of the function

    Returns:
        str: Path to the cache file
    """
    safe_model = model.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe_model}__{name}.json")


def analyze_function(name, config):
    """Analyze a function using the configured model

    Cached results are reused when available

    Args:
        name (str): name of the function to analyze
        config (RunConfig): Application configuration

    Returns:
        FunctionProfile: Analyzed function profile
    """
    path = _cache_path(config.model, name)
    if os.path.exists(path):
        with open(path) as fh:
            return _parse_profile(name, json.load(fh))  # cache hit, skip llm

    raw = _chat_json(config.model, PROMPT_TEMPLATE.format(name=name))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(raw, fh, indent=2)

    return _parse_profile(name, raw)
