// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace testing;
using namespace ov;

namespace {

struct SchedulingCoreTypeTestCase {
    ov::hint::SchedulingCoreType input_type;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> result_table;
    ov::hint::SchedulingCoreType output_type;
};

class SchedulingCoreTypeTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<SchedulingCoreTypeTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());
        auto test_input_type = test_data.input_type;

        std::vector<std::vector<int>> test_result_table =
            ov::intel_cpu::apply_scheduling_core_type(test_input_type, test_data.proc_type_table);

        ASSERT_EQ(test_data.result_table, test_result_table);
        ASSERT_EQ(test_data.output_type, test_input_type);
    }
};

SchedulingCoreTypeTestCase _2sockets_ALL = {
    ov::hint::SchedulingCoreType::ANY_CORE,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::ANY_CORE,
};

SchedulingCoreTypeTestCase _2sockets_P_CORE_ONLY = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
};

SchedulingCoreTypeTestCase _2sockets_E_CORE_ONLY = {
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    // ov::hint::scheduling_core_type returns ANY_CORE because the platform has no Ecores available to satisfy the
    // user's request.
};

SchedulingCoreTypeTestCase _1sockets_ALL = {
    ov::hint::SchedulingCoreType::ANY_CORE,
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    ov::hint::SchedulingCoreType::ANY_CORE,
};

SchedulingCoreTypeTestCase _1sockets_P_CORE_ONLY = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{20, 6, 8, 6}},
    {{12, 6, 0, 6}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
};

SchedulingCoreTypeTestCase _1sockets_P_CORE_ONLY_1 = {
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    {{8, 0, 8, 0}},
    {{8, 0, 8, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    // ov::hint::scheduling_core_type returns ANY_CORE because the platform has no Pcore available to satisfy the
    // user's request.
};

SchedulingCoreTypeTestCase _1sockets_E_CORE_ONLY = {
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    {{20, 6, 8, 6}},
    {{8, 0, 8, 0}},
    ov::hint::SchedulingCoreType::ECORE_ONLY,
};

TEST_P(SchedulingCoreTypeTests, SchedulingCoreType) {}

INSTANTIATE_TEST_SUITE_P(SchedulingCoreTypeTable,
                         SchedulingCoreTypeTests,
                         testing::Values(_2sockets_ALL,
                                         _2sockets_P_CORE_ONLY,
                                         _2sockets_E_CORE_ONLY,
                                         _1sockets_ALL,
                                         _1sockets_P_CORE_ONLY,
                                         _1sockets_P_CORE_ONLY_1,
                                         _1sockets_E_CORE_ONLY));
}  // namespace