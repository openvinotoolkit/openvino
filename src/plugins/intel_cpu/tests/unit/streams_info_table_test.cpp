// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_system_conf.h>

#include <common_test_utils/test_common.hpp>

#include "cpu_streams_calculation.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace ov;

namespace {

struct ProcessorTypeTestCase {
    ov::intel_cpu::ProcessorType input_type;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> result_table;
};

class ProcessorTypeTests : public CommonTestUtils::TestsCommon,
                           public testing::WithParamInterface<std::tuple<ProcessorTypeTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_result_table =
            ov::intel_cpu::apply_processor_type(test_data.input_type, test_data.proc_type_table);

        ASSERT_EQ(test_data.result_table, test_result_table);
    }
};

ProcessorTypeTestCase _2sockets_UNDEFINED = {
    ov::intel_cpu::ProcessorType::UNDEFINED,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
};

ProcessorTypeTestCase _2sockets_ALL = {
    ov::intel_cpu::ProcessorType::ALL,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
};

ProcessorTypeTestCase _2sockets_PHY_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::PHY_CORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
};

ProcessorTypeTestCase _2sockets_P_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::P_CORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
};

ProcessorTypeTestCase _2sockets_E_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::E_CORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
};

ProcessorTypeTestCase _2sockets_PHY_P_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::PHY_P_CORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
};

ProcessorTypeTestCase _1sockets_UNDEFINED = {
    ov::intel_cpu::ProcessorType::UNDEFINED,
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
};

ProcessorTypeTestCase _1sockets_ALL = {
    ov::intel_cpu::ProcessorType::ALL,
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
};

ProcessorTypeTestCase _1sockets_PHY_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::PHY_CORE_ONLY,
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
};

ProcessorTypeTestCase _1sockets_P_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::P_CORE_ONLY,
    {{20, 6, 8, 6}},
    {{12, 6, 0, 6}},
};

ProcessorTypeTestCase _1sockets_E_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::E_CORE_ONLY,
    {{20, 6, 8, 6}},
    {{8, 0, 8, 0}},
};

ProcessorTypeTestCase _1sockets_PHY_P_CORE_ONLY = {
    ov::intel_cpu::ProcessorType::PHY_P_CORE_ONLY,
    {{20, 6, 8, 6}},
    {{6, 6, 0, 0}},
};

TEST_P(ProcessorTypeTests, ProcessorType) {}

INSTANTIATE_TEST_SUITE_P(ProcessorTypeTable,
                         ProcessorTypeTests,
                         testing::Values(_2sockets_UNDEFINED,
                                         _2sockets_ALL,
                                         _2sockets_PHY_CORE_ONLY,
                                         _2sockets_P_CORE_ONLY,
                                         _2sockets_E_CORE_ONLY,
                                         _2sockets_PHY_P_CORE_ONLY,
                                         _1sockets_UNDEFINED,
                                         _1sockets_ALL,
                                         _1sockets_PHY_CORE_ONLY,
                                         _1sockets_P_CORE_ONLY,
                                         _1sockets_E_CORE_ONLY,
                                         _1sockets_PHY_P_CORE_ONLY));

}  // namespace