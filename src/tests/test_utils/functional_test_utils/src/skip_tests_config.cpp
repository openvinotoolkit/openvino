// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <fstream>
#include <iostream>
#include <unordered_set>

#include "common_test_utils/file_utils.hpp"

namespace ov::test::utils {

bool disable_tests_skipping = false;
namespace {

bool precomputed_disabled_tests_enabled = false;
std::unordered_set<std::string> precomputed_disabled_tests;

inline std::string get_current_test_name(const ::testing::TestInfo* test_info) {
    std::string full_name;
    full_name.reserve(std::strlen(test_info->test_case_name()) + 1 + std::strlen(test_info->name()));
    full_name.append(test_info->test_case_name());
    full_name.push_back('.');
    full_name.append(test_info->name());

    return full_name;
}

}  // namespace

void set_disabled_tests_filter_from_patterns() {
    if (disable_tests_skipping) {
        return;
    }

    const auto& patterns = disabled_test_patterns();
    if (patterns.empty()) {
        return;
    }

    auto* unit_test = ::testing::UnitTest::GetInstance();
    std::unordered_set<std::string> disabled_tests;

    for (int suite_idx = 0; suite_idx < unit_test->total_test_suite_count(); ++suite_idx) {
        const auto* test_suite = unit_test->GetTestSuite(suite_idx);
        if (test_suite == nullptr) {
            continue;
        }

        for (int test_idx = 0; test_idx < test_suite->total_test_count(); ++test_idx) {
            const auto* test_info = test_suite->GetTestInfo(test_idx);
            if (test_info == nullptr) {
                continue;
            }

            auto full_name = get_current_test_name(test_info);

            for (const auto& re : patterns) {
                if (std::regex_match(full_name, re)) {
                    disabled_tests.emplace(std::move(full_name));
                    break;
                }
            }
        }
    }

    precomputed_disabled_tests = std::move(disabled_tests);
    precomputed_disabled_tests_enabled = true;
}

bool current_test_is_disabled() {
    if (disable_tests_skipping) {
        return false;
    }

    const auto* current_test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (current_test_info == nullptr) {
        return false;
    }

    auto full_name = get_current_test_name(current_test_info);

    if (precomputed_disabled_tests_enabled) {
        return precomputed_disabled_tests.find(full_name) != precomputed_disabled_tests.end();
    }

    for (const auto& re : disabled_test_patterns()) {
        if (std::regex_match(full_name, re)) {
            return true;
        }
    }

    return false;
}

}  // namespace ov::test::utils
