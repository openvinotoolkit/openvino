// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace tests {

// Parameterized test name generators must not emit characters that are part of the
// gtest filter syntax ('-', ':', '*', '?'): the GPU CI test runner (gtest_parallel.py)
// executes every test as a separate process with `--gtest_filter=<full name>`, and
// gtest splits such a filter ('-' into positive/negative parts, ':' into patterns),
// so the name matches 0 tests, the process exits 0 and the runner records a passing
// run that executed nothing.
// Replacements keep names deterministic, collision-free and readable:
//   '-' -> "neg" (negative values), ':' -> '_' (tensor/layout descriptions),
//   '*' -> "star", '?' -> "dyn" (dynamic dimensions).
inline std::string sanitize_test_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    for (char c : name) {
        switch (c) {
        case '-':
            result += "neg";
            break;
        case ':':
            result += '_';
            break;
        case '*':
            result += "star";
            break;
        case '?':
            result += "dyn";
            break;
        default:
            result += c;
            break;
        }
    }
    return result;
}

}  // namespace tests
