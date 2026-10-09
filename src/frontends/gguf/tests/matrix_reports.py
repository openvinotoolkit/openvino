# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate GTest and GGUF GenAI report contracts and results."""

import json
import math
import xml.etree.ElementTree as ET


# GTest unit and integration reports


def validate_gtest_contract(name, report):
    count = report.get("expected_tests")
    skips = report.get("max_skips", 0)
    if type(count) is not int or count <= 0 or type(skips) is not int or skips < 0:
        raise ValueError(f"{name}: gtest requires positive expected_tests and nonnegative max_skips")


def audit_gtest_report(contract, path):
    tests = list(ET.parse(path).getroot().iter("testcase"))
    skipped = sum(test.find("skipped") is not None or test.get("status") == "notrun" for test in tests)
    failures = sum(test.find("failure") is not None or test.find("error") is not None for test in tests)
    if len(tests) != contract["expected_tests"] or skipped > contract.get("max_skips", 0) or failures:
        raise ValueError(f"gtest: tests={len(tests)}, skips={skipped}, failures={failures}")
    return {"tests": len(tests), "skips": skipped, "failures": failures}


# GGUF GenAI accuracy and API reports


def validate_gguf_mmproj_contract(name, report):
    fraction = report.get("min_choice_fraction")
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError(f"{name}: gguf_mmproj requires min_choice_fraction in [0, 1]")
    if type(report.get("first_token_matches")) is not bool:
        raise ValueError(f"{name}: gguf_mmproj requires explicit first_token_matches")
    for key in ("required_modalities", "required_api_checks"):
        values = report.get(key, [])
        if (not isinstance(values, list) or not all(isinstance(v, str) and v for v in values) or
                len(set(values)) != len(values) or (key == "required_modalities" and not values)):
            raise ValueError(f"{name}: invalid {key}")
    if not isinstance(report.get("equals", {}), dict):
        raise ValueError(f"{name}: equals must be an object of required top-level report values")


def audit_gguf_mmproj_report(contract, path):
    report = json.loads(path.read_text())
    if report.get("completed") is not True:
        raise ValueError("GGUF report did not complete")
    for key, value in contract.get("equals", {}).items():
        if key not in report or report[key] != value:
            raise ValueError(f"GGUF report {key} differs from the contract")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("GGUF report has no modality cases")
    modalities = [case["modality"] for case in cases]
    if len(set(modalities)) != len(modalities) or set(contract["required_modalities"]) - set(modalities):
        raise ValueError(f"GGUF modalities missing or duplicated: {modalities}")
    fractions = []
    for case in cases:
        tokens, choices = case["tokens"], case["reference_choices_on_same_history"]
        if (not isinstance(tokens, list) or not isinstance(choices, list) or not tokens or
                len(tokens) != len(choices) or any(type(t) is not int for t in tokens + choices)):
            raise ValueError(f"GGUF {case['modality']}: missing or unequal token histories")
        first = tokens[0] == choices[0]
        fraction = sum(a == b for a, b in zip(tokens, choices)) / len(tokens)
        reported_fraction = case.get("matching_choice_fraction")
        if (case.get("first_token_matches") is not first or isinstance(reported_fraction, bool) or
                not isinstance(reported_fraction, (int, float)) or not math.isfinite(reported_fraction) or
                not math.isclose(reported_fraction, fraction, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"GGUF {case['modality']}: reported scores disagree with token histories")
        if (contract["first_token_matches"] and not first) or fraction < contract["min_choice_fraction"]:
            raise ValueError(f"GGUF {case['modality']}: first_token={first}, choice_fraction={fraction}")
        fractions.append(fraction)
    checks = report.get("api_checks", {})
    if not isinstance(checks, dict) or set(contract.get("required_api_checks", [])) - set(checks):
        raise ValueError("GGUF API checks missing")
    if any(check.get("passed") is not True for check in checks.values()):
        raise ValueError("GGUF API checks failed")
    if report.get("passed") is not True:
        raise ValueError("GGUF report did not pass")
    return {"modalities": modalities, "minimum_choice_fraction": min(fractions), "api_checks": list(checks)}


# Report dispatch


def audit_report(contract, path):
    if contract["kind"] == "gtest":
        return audit_gtest_report(contract, path)
    return audit_gguf_mmproj_report(contract, path)


