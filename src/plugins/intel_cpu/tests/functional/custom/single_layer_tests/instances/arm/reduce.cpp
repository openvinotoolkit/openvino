// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/reduce.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Reduce {
namespace {

std::vector<std::vector<ov::test::InputShape>> inputShapes_5D = {
    {{{}, {{2, 19, 2, 2, 9}}}},
};

const std::vector<std::vector<int>> axes5D = {
        {2, 4},
        {1, 2, 4},
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
};

const auto params_MultiAxis_5D = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(additionalConfig()));

const std::vector<std::vector<int>> axes5D_ref = {
        {0}
};

std::vector<CPUSpecificParams> cpuParams_5D_ref = {
        CPUSpecificParams({ncdhw}, {ncdhw}, {"ref"}, {"ref"}),
};

std::vector<std::map<std::string, ov::element::Type>> config_infer_prec_f32 = {
        {{ov::hint::inference_precision.name(), ov::element::f32}}
    };

const auto params_MultiAxis_5D_ref = testing::Combine(
        testing::Combine(
                testing::ValuesIn(axes5D_ref),
                testing::Values(ov::test::utils::OpType::VECTOR),
                testing::Values(true),
                testing::ValuesIn(reductionTypes()),
                testing::ValuesIn(inpOutPrc()),
                testing::Values(ElementType::undefined),
                testing::Values(ElementType::undefined),
                testing::ValuesIn(inputShapes_5D)),
        testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_ref)),
        testing::Values(emptyFusingSpec),
        testing::ValuesIn(config_infer_prec_f32));

//There are dedicated instences of smoke_Reduce_MultiAxis_5D_CPU test in arm and x64 folders
//because ACL does not support 0 as reduction axis
INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_CPU,
        ReduceCPULayerTest,
        params_MultiAxis_5D,
        ReduceCPULayerTest::getTestCaseName
);

// Reference implementation testing of ACL unsupported case
INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_CPU_ref,
        ReduceCPULayerTest,
        params_MultiAxis_5D_ref,
        ReduceCPULayerTest::getTestCaseName
);

}  // namespace
}  // namespace Reduce
}  // namespace test
}  // namespace ov
