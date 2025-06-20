"""
Data Science Team Notebook - Academic Integrity Framework - Major Verification Module
-------------------------------------------------------------------------------------
Author(s): Naveen N
Date: 2025-05-05
Version: 2.0

Module Description:
    This module implements a major verification pipeline using probabilistic scoring based on 
    university data, large language models (LLMs), and real-time search context via Tavily API.

    The module verifies if a claimed major at a specific university during a particular year is likely to be genuine.
    It combines LLM reasoning with live web evidence retrieved using Tavily to evaluate:
        - Major existence in archived catalogs
        - Institution accreditation and reputation
        - Major-university consistency
        - University acceptance rate for that year
        - Major-specific acceptance rate
        - GPA-based admission probability

    The final output is a confidence score (0.0–1.0) for each major claim, used to flag likely fraudulent entries.

    The module can be divided into the following major steps:
        1. Extraction of unverified major claims from the certification record
        2. Prompt generation for each verification dimension
        3. Real-time web search using Tavily for contextual evidence
        4. Prompt injection and LLM-based scoring using GPT-4
        5. Probabilistic scoring based on weighted dimensions
        6. Update and return of enriched input JSON structure with detailed confidence values inside major fields

Assumptions:
    - The input follows the `example_certification` schema with nested `education` records.
    - Majors labeled as "MATCH_TBD" are the only ones processed.
    - Dates are expected in the format: "Month DD, YYYY", "Mon DD, YYYY", or "YYYY".
    - Output is a soft verification confidence score, not a binary decision.
    - Requires valid API keys for OpenAI and Tavily stored in environment variables.

Dependencies:
    - Python version: 3.10+
    - Libraries:
        - `openai>=1.0`
        - `tavily-python`
        - `python-dotenv`
        - `re`, `json`, `datetime`

File Structure:
    - major_verification_pipeline_v1.py: This module
    - .env: Contains API keys for OpenAI and Tavily
    - data/: Input and test samples (optional)
    - results/: Verified outputs with scores (optional)

Overview of Major Sections:
    1. **Score Extraction Logic**:
        - `call_llm_with_tavily`: Sends prompt with Tavily search context to OpenAI and extracts float score.

    2. **Prompt Generators**:
        - Six different scoring functions:
            - `generate_prompt_major_existence`
            - `generate_prompt_institution_reputation`
            - `generate_prompt_major_school_consistency`
            - `generate_prompt_acceptance_rate`
            - `generate_prompt_major_acceptance_rate`
            - `generate_prompt_gpa_admission_probability`

    3. **Pipeline Execution**:
        - `update_major_scores`:
            - Iterates through all education entries
            - Processes only majors with `comparison_result: MATCH_TBD`
            - Applies all six prompts with search context
            - Injects `confidence_score` and `meta` dictionary into the respective major entry
            - Returns the enriched JSON

    4. **Scoring Logic**:
        - `calculate_final_score`:
            Uses empirically backed weights:
                - 40% Major Existence
                - 20% Institution Reputation
                - 10% Major-School Fit
                - 10% University Acceptance Rate
                - 15% Major-Specific Acceptance Rate
                - 5%  GPA Admission Probability

Usage Example:
    from major_verification_pipeline_v2 import update_major_scores

    example_certification = {
        "person_of_interest": {
            "education": [
                {
                    "school_name": {"value": "Stanford University"},
                    "dates_of_attendance": {
                        "start_date": {"value": "August 15, 2010"},
                        "end_date": {"value": "May 30, 2014"}
                    },
                    "major": [
                        {"value": "Computer Science", "comparison_result": "MATCH_TBD"}
                    ]
                }
            ]
        }
    }

    updated_certification = update_major_scores(example_certification)

    import json
    print(json.dumps(updated_certification, indent=4))
"""


import json
import re
import os
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

client = OpenAI(api_key='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')
tavily_client = TavilyClient(api_key = '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')

# ----------------------------- LLM CALL WITH TAVILY SEARCH -----------------------------

def call_llm_for_score_with_tavily(prompt_dict):
    search_query = prompt_dict['task']
    try:
        search_results = tavily_client.search(query=search_query, max_results=5)
        sources = "\n\n".join([item['content'] for item in search_results['results']])

        mega_prompt = f"""
You are an academic verification agent using web evidence and domain knowledge.

System Instruction:
{prompt_dict['system']}

Search Context (from web):
{sources}

Task:
{prompt_dict['task']}

Instruction:
{prompt_dict['instruction']}

Only respond with a float score between 0.0 and 1.0. No explanation.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You verify academic records based on web search evidence and domain rules."},
                {"role": "user", "content": mega_prompt}
            ],
            temperature=0,
            max_tokens=10,
        )

        raw_output = response.choices[0].message.content.strip()
        score_match = re.search(r"[01](?:\.\d+)?", raw_output)
        if score_match:
            return float(score_match.group(0))
        else:
            print(f"[Error] Could not extract score from response: {raw_output}")
            return 0.0

    except Exception as e:
        print(f"[Error] Tavily + LLM call failed: {e}")
        return 0.0

# ----------------------------- PROMPT GENERATORS -----------------------------

def generate_prompt_major_existence(university, major, year):
    return {
        "system": "You are an expert in academic archives and catalogs.",
        "task": f"Did '{university}' offer the major '{major}' in {year}?",
        "instruction": "Return a single numeric confidence score as a float between 0.0 and 1.0, based on the strength of available evidence. Do not include any explanation or text—only the float value."
    }

def generate_prompt_institution_reputation(university):
    return {
        "system": "You analyze global university reputation and accreditation.",
        "task": f"Assess the accreditation and reputation of '{university}'.",
        "instruction": "Based on institutional rankings and accreditation status, provide a confidence score as a float between 0.0 and 1.0 reflecting the university’s overall reputation. Return only the numeric value. Do not include any explanation or additional text."
    }

def generate_prompt_major_school_consistency(university, major, year):
    return {
        "system": "You know which programs align with each university’s academic strengths.",
        "task": f"Evaluate how consistent the major '{major}' is with '{university}' academic focus in {year}.",
        "instruction": "Provide a confidence score as a float between 0.0 and 1.0 indicating how well the major aligns with the university’s academic strengths. Return only the float value, no explanation or additional text."
    }

def generate_prompt_acceptance_rate(university, year):
    return {
        "system": "You provide historical university admission rates.",
        "task": f"What was the acceptance rate for '{university}' in {year}?",
        "instruction": "Return only a float (e.g., 0.23 for 23%). Use best estimate."
    }

def generate_prompt_major_acceptance_rate(university, major, year):
    return {
        "system": "You analyze program-level admissions statistics.",
        "task": f"What was the acceptance rate for major '{major}' at '{university}' in {year}?",
        "instruction": "Provide an estimated acceptance rate as a float between 0.0 and 1.0 based on available data. Return only the numeric value without any explanation or text."
    }

def generate_prompt_gpa_admission_probability(university, major, year):
    return {
        "system": "You estimate GPA-based admission probability.",
        "task": f"What was the probability that an average student got into '{major}' at '{university}' in {year}?",
        "instruction": "Return only a float score from 0.0 to 1.0."
    }

# ----------------------------- SCORING & VERIFICATION -----------------------------

def calculate_final_score(existence, reputation, consistency, uni_acceptance, major_acceptance, gpa_score):
    return (
        (0.40 * existence) +
        (0.20 * reputation) +
        (0.10 * consistency) +
        (0.10 * uni_acceptance) +
        (0.15 * major_acceptance) +
        (0.05 * gpa_score)
    )

def verify_major_claim(university, major, year):
    prompts = {
        "existence": generate_prompt_major_existence(university, major, year),
        "reputation": generate_prompt_institution_reputation(university),
        "consistency": generate_prompt_major_school_consistency(university, major, year),
        "uni_acceptance_rate": generate_prompt_acceptance_rate(university, year),
        "major_acceptance_rate": generate_prompt_major_acceptance_rate(university, major, year),
        "major_admit_gpa_prob": generate_prompt_gpa_admission_probability(university, major, year)
    }

    scores = {k: call_llm_for_score_with_tavily(p) for k, p in prompts.items()}

    final_score = calculate_final_score(
        scores["existence"],
        scores["reputation"],
        scores["consistency"],
        scores["uni_acceptance_rate"],
        scores["major_acceptance_rate"],
        scores["major_admit_gpa_prob"]
    )

    return {
        "confidence_score": final_score,
        "meta": scores
    }
def update_major_scores(example_certification):
    for edu_record in example_certification.get("person_of_interest", {}).get("education", []):
        university = edu_record.get("school_name", {}).get("value", "")
        end_date = edu_record.get("dates_of_attendance", {}).get("end_date", {}).get("value", "")

        try:
            year = datetime.strptime(end_date, "%B %d, %Y").year
        except ValueError:
            try:
                year = datetime.strptime(end_date, "%b %d, %Y").year
            except ValueError:
                year = None

        if "major" in edu_record:
            for major in edu_record["major"]:
                if major.get("comparison_result") == "MATCH_TBD":
                    major_value = major.get("value", "UNKNOWN")
                    verification = verify_major_claim(university, major_value, year)
                    major["confidence_score"] = verification["confidence_score"]
                    major["meta"] = verification["meta"]

    return example_certification
