// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <array>
#include <map>
#include <string>

namespace ov::test::utils {

enum class VpuTestStage { RUN = 0, COMPILED, IMPORTED, INFERRED, VALIDATED, SKIPPED_EXCEPTION, LAST_VALUE };

class VpuTestReport {
private:
    std::map<std::string, std::array<int, static_cast<int>(VpuTestStage::LAST_VALUE)>> counters;

public:
    static const std::array<std::string, static_cast<int>(VpuTestStage::LAST_VALUE)> stages;

public:
    explicit VpuTestReport();
    void run(const testing::TestInfo* testInfo);
    void compiled(const testing::TestInfo* testInfo);
    void imported(const testing::TestInfo* testInfo);
    void inferred(const testing::TestInfo* testInfo);
    void validated(const testing::TestInfo* testInfo);
    void skipped(const testing::TestInfo* testInfo);

    const std::map<std::string, std::array<int, static_cast<int>(VpuTestStage::LAST_VALUE)>>& getCounters() const {
        return counters;
    }

    static VpuTestReport& getInstance() {
        static VpuTestReport instance;
        return instance;
    }
};

class NpuTestReportEnvironment : public testing::Environment {
public:
    ~NpuTestReportEnvironment() override {
    }

    void TearDown() override;
};

}  // namespace ov::test::utils
