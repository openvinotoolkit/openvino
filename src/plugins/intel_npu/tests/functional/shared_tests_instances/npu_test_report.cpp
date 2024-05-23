// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_test_report.hpp"
#include "functional_test_utils/summary/op_summary.hpp"

#include <fstream>
#include <iostream>

namespace ov::test::utils {

const std::array<std::string, static_cast<int>(VpuTestStage::LAST_VALUE)> VpuTestReport::stages = {
        "RUN", "COMPILED", "IMPORTED", "INFERRED", "VALIDATED", "SKIPPED_EXCEPTION"};

std::string testName(const testing::TestInfo* testInfo);

std::string testName(const testing::TestInfo* testInfo) {
    const std::string name(testInfo->test_case_name());
    auto npos = name.find("/");

    return (npos != std::string::npos) ? name.substr(npos + 1) : name;
}

VpuTestReport::VpuTestReport() {
}

void VpuTestReport::run(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::RUN)];
}

void VpuTestReport::compiled(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::COMPILED)];
}

void VpuTestReport::imported(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::IMPORTED)];
}

void VpuTestReport::inferred(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::INFERRED)];
}

void VpuTestReport::validated(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::VALIDATED)];
}

void VpuTestReport::skipped(const testing::TestInfo* testInfo) {
    ++counters[testName(testInfo)][static_cast<int>(VpuTestStage::SKIPPED_EXCEPTION)];
}

void NpuTestReportEnvironment::TearDown() {
    std::cout << "TestReportResult: " << std::endl;
    const auto& counters = VpuTestReport::getInstance().getCounters();

    std::array<int, static_cast<int>(VpuTestStage::LAST_VALUE)> totals = {};
    for (auto const& cit : counters) {
        std::cout << cit.first << ": ";
        for (int it = static_cast<int>(VpuTestStage::RUN); it < static_cast<int>(VpuTestStage::LAST_VALUE); ++it) {
            totals[it] += cit.second[it];
            std::cout << VpuTestReport::stages[it] << " - " << cit.second[it] << "; ";
        }
        std::cout << std::endl;
    }

    std::cout << "VpuTotalTestCases: ";
    for (int it = static_cast<int>(VpuTestStage::RUN); it < static_cast<int>(VpuTestStage::LAST_VALUE); ++it) {
        std::cout << VpuTestReport::stages[it] << " - " << totals[it] << "; ";
    }
    std::cout << std::endl;

    ov::test::utils::OpSummary::getInstance().saveReport();
};

}  // namespace ov::test::utils
