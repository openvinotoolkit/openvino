// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/pooling.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Pooling {

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_2x2kernel = {
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER }
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_2x2Kernel, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(filterCPUInfo(vecCpuConfigsFusing_4D())),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_non2x2kernel = {
            maxPoolV8SpecificParams{ {11, 7}, {2, 2}, {1, 1}, {2, 2}, {2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT},
};

//The test checks that fallback to nGraph works for ACL non-supported cases
INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_non2x2Kernel_ref, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_non2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

}  // namespace Pooling
}  // namespace test
}  // namespace ov