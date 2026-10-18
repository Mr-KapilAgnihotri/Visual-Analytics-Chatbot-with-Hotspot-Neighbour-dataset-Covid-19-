"""
Shared test setup.

Tests must give the same result on every run and every machine, so they always
use the rule-based path: the language model is switched off here, whatever the
.env file says. (The LLM layer is still tested, with a fake LLM, in
test_pipeline.py::test_llm_merge_* .)
This file is loaded by pytest before any test module imports the backend.
"""
import os

os.environ["USE_LLM"] = "false"
