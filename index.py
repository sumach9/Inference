from inference_institution_stat_v0 import get_institution_stats
from inference_honors_academichonors import full_verification_pipeline
from get_date_confidence_tbd_score import get_date_confidence_tbd_score
from major_verification_pipeline_v1 import update_major_scores
from inference_minor_v2 import verify_and_annotate # supress all outputs #Suma
from inference_degree_stat_v1 import validate_degrees # supress all outputs #Subha


# Re-import necessary packages after reset
import pandas as pd

# Constants
BASE_SCORE = 0
ALPHA = 0.5
GAMMA = 0.2

# Sample input
example_certification ={
        "person_of_interest": {
            "last_name": {
                "extracted_value": "Smithereens",
                "sot_value": "SMITHEREENS",
                "comparison_result": "MATCH"
            },
            "first_name": {
                "extracted_value": "John",
                "sot_value": "JOHN",
                "comparison_result": "MATCH"
            },
            "education": [
                {
                    "school_name": {
                        "extracted_value": "VS Hometown University",
                        "sot_value": "VS - HOMETOWN UNIVERSITY",
                        "comparison_result": "MATCH"
                    },
                    "degree_title": {
                        "extracted_value": "Master of Science",
                        "sot_value": "MASTER OF SCIENCE",
                        "comparison_result": "MATCH"
                    },
                    "date_of_degree_awarded": {
                        "extracted_value": "1997",
                        "sot_value": "2015-05-31",
                        "comparison_result": "NO_MATCH"
                    },
                    "honors_program": {
                        "extracted_value": "",
                        "sot_value": "1988 ACADEMIC AWARD",
                        "comparison_result": "MATCH_NA"
                    },
                    "major": [
                        {
                            "extracted_value": "Engineering",
                            "sot_value": "ENGINEERING",
                            "comparison_result": "MATCH"
                        },
                        {
                            "extracted_value": "Mathematics",
                            "sot_value": "MATHEMATICS AND STATISTICS",
                            "comparison_result": "MATCH"
                        }
                    ],
                    "minor": [
                        {
                            "extracted_value": "",
                            "sot_value": "COMPUTER SCIENCE",
                            "comparison_result": "MATCH_NA"
                        },
                        {
                            "extracted_value": "",
                            "sot_value": "PHILOSOPHY",
                            "comparison_result": "MATCH_NA"
                        },
                        {
                            "extracted_value": "",
                            "sot_value": "MATHEMATICS",
                            "comparison_result": "MATCH_NA"
                        },
                        {
                            "extracted_value": "",
                            "sot_value": "LINGUISTICS",
                            "comparison_result": "MATCH_NA"
                        }
                    ],
                    "academic_honors": [
                        {
                            "extracted_value": "",
                            "sot_value": "MAGNA CUM LAUDE",
                            "comparison_result": "MATCH_NA"
                        }
                    ],
                    "dates_of_attendance": {
                        "start_date": {
                            "extracted_value": "",
                            "sot_value": "2011-07-31",
                            "comparison_result": "MATCH_NA"
                        },
                        "end_date": {
                            "extracted_value": "1997",
                            "sot_value": "2015-04-30",
                            "comparison_result": "NO_MATCH"
                        }
                    }
                },
                {
                    "school_name": {
                        "extracted_value": "VS Hometown University",
                        "sot_value": "VS - HOMETOWN UNIVERSITY",
                        "comparison_result": "MATCH"
                    },
                    "degree_title": {
                        "extracted_value": "BS",
                        "sot_value": "BACHELOR OF SCIENCE",
                        "comparison_result": "MATCH"
                    },
                    "date_of_degree_awarded": {
                        "extracted_value": "1995",
                        "sot_value": "2015-05-31",
                        "comparison_result": "NO_MATCH"
                    },
                    "honors_program": {
                        "extracted_value": "",
                        "sot_value": "",
                        "comparison_result": "MATCH_NA"
                    },
                    "major": [
                        {
                            "extracted_value": "Engineering",
                            "sot_value": "",
                            "comparison_result": "MATCH_TBD"
                        }
                    ],
                    "minor": [
                        {
                            "extracted_value": "Computer Science",
                            "sot_value": "",
                            "comparison_result": "MATCH_TBD"
                        }
                    ],
                    "academic_honors": [
                        {
                            "extracted_value": "",
                            "sot_value": "",
                            "comparison_result": "MATCH_NA"
                        }
                    ],
                    "dates_of_attendance": {
                        "start_date": {
                            "extracted_value": "",
                            "sot_value": "2011-07-31",
                            "comparison_result": "MATCH_NA"
                        },
                        "end_date": {
                            "extracted_value": "1995",
                            "sot_value": "2015-04-30",
                            "comparison_result": "NO_MATCH"
                        }
                    }
                }
            ]
        }
    }
ACCEPTANCE_RATE_MAP = get_institution_stats(example_certification)

# Utilities
def compute_percent_fields_match(block, exclude_field_key=None):
    total = 0
    match = 0
    for key, value in block.items():
        if key == exclude_field_key:
            continue
        if isinstance(value, dict) and 'comparison_result' in value:
            if value['comparison_result'] != 'MATCH_NA':
                total += 1
                if value['comparison_result'] == 'MATCH':
                    match += 1
        elif isinstance(value, list):
            for item in value:
                if 'comparison_result' in item and item['comparison_result'] != 'MATCH_NA':
                    total += 1
                    if item['comparison_result'] == 'MATCH':
                        match += 1
    return match / total if total > 0 else 0

# Individual scoring functions


def score_school_name(value, context, acceptance_rate):
    #return 0.3
    score =  max(0, BASE_SCORE + (ALPHA * context) - (GAMMA * (1 - acceptance_rate)))
    return score


# Dispatch table
field_scoring_functions = {

    "school_name": score_school_name,
}

# Wrapper function
import copy

# Function to iterate and update JSON with confidence scores using the wrapper function
def iterate_and_update_json(data):
    updated_data = copy.deepcopy(data)
    education_blocks = updated_data["person_of_interest"]["education"]

    for edu in education_blocks:
        school_name = edu.get("school_name", {}).get("value", "")
        for key, value in edu.items():
            if key == 'school_name' :

              context_score = compute_percent_fields_match(edu, exclude_field_key=key)
              acceptance_rate = ACCEPTANCE_RATE_MAP.get(school_name, 0.0)
              # acceptance_rate = 0
              score_fn = field_scoring_functions.get(key)

              if isinstance(value, dict) and value.get("comparison_result") == "MATCH_TBD" and score_fn:
                  score = score_fn(value.get("value"), context_score, acceptance_rate)
                  print(score)
                  edu[key]["confidence_score"] = round(min(score or 0, 0.99), 2)
                  edu[key]['acceptance_rate'] = acceptance_rate
                  edu[key]['context_score'] = context_score

              elif isinstance(value, list) and score_fn:
                  for item in value:
                      if item.get("comparison_result") == "MATCH_TBD":
                          score = score_fn(item.get("value"), context_score, acceptance_rate)
                          item["confidence_score"] = round(min(score or 0, 0.99), 2)
                          item["context_score"] = context_score

    return updated_data

# Run the iteration and update
institution_confidence = iterate_and_update_json(example_certification)

date_confidence = get_date_confidence_tbd_score(institution_confidence)
degree_confidence = validate_degrees(date_confidence)
minor_confidence = verify_and_annotate(degree_confidence)
score , honors_confidence = full_verification_pipeline(minor_confidence)
major_confidence = update_major_scores(honors_confidence)