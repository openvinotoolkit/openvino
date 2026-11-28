// Copyright (C) 2018-2026 Intel Corporation
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

std::vector<std::vector<ov::test::InputShape>> inputShapes_5D_ZeroDim = {
    {{{}, {{2, 19, 0, 2, 9}}}},
    {{{}, {{2, 19, 0, 2, 0}}}},
};

std::vector<std::vector<ov::test::InputShape>> inputShapes_EmptyAxes = {
    {{{}, {{2, 19, 2, 9}}}},
};

const std::vector<std::vector<int>> axes5D = {
        {2, 4},
        {1, 2, 4},
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {}),
};

const auto params_MultiAxis_5D = testing::Combine(testing::Combine(testing::ValuesIn(axes5D),
                                                                   testing::Values(ov::test::utils::OpType::VECTOR),
                                                                   testing::Values(true),
                                                                   testing::ValuesIn(reductionTypes()),
                                                                   testing::ValuesIn(inpOutPrc()),
                                                                   testing::Values(ElementType::dynamic),
                                                                   testing::Values(ElementType::dynamic),
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

const auto params_MultiAxis_5D_ref = testing::Combine(testing::Combine(testing::ValuesIn(axes5D_ref),
                                                                       testing::Values(ov::test::utils::OpType::VECTOR),
                                                                       testing::Values(true),
                                                                       testing::ValuesIn(reductionTypes()),
                                                                       testing::ValuesIn(inpOutPrc()),
                                                                       testing::Values(ElementType::dynamic),
                                                                       testing::Values(ElementType::dynamic),
                                                                       testing::ValuesIn(inputShapes_5D)),
                                                      testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_ref)),
                                                      testing::Values(emptyFusingSpec),
                                                      testing::ValuesIn(config_infer_prec_f32));

const auto params_MultiAxis_5D_ZeroDim_ref =
    testing::Combine(testing::Combine(testing::ValuesIn(axes5D),
                                      testing::Values(ov::test::utils::OpType::VECTOR),
                                      testing::Values(true),
                                      testing::ValuesIn(reductionTypes()),
                                      testing::ValuesIn(inpOutPrc()),
                                      testing::Values(ElementType::dynamic),
                                      testing::Values(ElementType::dynamic),
                                      testing::ValuesIn(inputShapes_5D_ZeroDim)),
                     testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_ref)),
                     testing::Values(emptyFusingSpec),
                     testing::ValuesIn(additionalConfigFP32()));

std::vector<CPUSpecificParams> cpuParams_4D_ref = {
        CPUSpecificParams({nchw}, {nchw}, {"ref"}, {"ref"}),
};

// An empty axes list reduces nothing, so the output keeps the shape of the input.
// ACL has no reduction primitive for such a list, so the reference implementation is
// used, and it is registered with the f32 precision only.
const auto params_EmptyAxes_ref =
    testing::Combine(testing::Combine(testing::Values(std::vector<int>{}),
                                      testing::Values(ov::test::utils::OpType::VECTOR),
                                      testing::Values(false),
                                      testing::ValuesIn(reductionTypes()),
                                      testing::ValuesIn(inpOutPrc()),
                                      testing::Values(ElementType::dynamic),
                                      testing::Values(ElementType::dynamic),
                                      testing::ValuesIn(inputShapes_EmptyAxes)),
                     testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_ref)),
                     testing::Values(emptyFusingSpec),
                     testing::ValuesIn(additionalConfigFP32()));

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

INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_MultiAxis_5D_ZeroDim_CPU_ref,
        ReduceCPULayerTest,
        params_MultiAxis_5D_ZeroDim_ref,
        ReduceCPULayerTest::getTestCaseName
);

// Reference implementation testing of ACL unsupported case
INSTANTIATE_TEST_SUITE_P(
        smoke_Reduce_EmptyAxes_CPU_ref,
        ReduceCPULayerTest,
        params_EmptyAxes_ref,
        ReduceCPULayerTest::getTestCaseName
);

// A long reduction axis under f16. The inputs are integers in [0, 10), exact in f16, while the partial sums exceed
// 2048 where the f16 step is 2. ACL splits the work over the non-reduced axis, and slices narrower than 8 elements
// take a scalar loop that already accumulates in f32: with a last axis of 1024, each thread gets ceil(1024 / threads)
// >= 8 elements, so the f16 vector loop runs with up to 146 threads.
std::vector<std::vector<ov::test::InputShape>> inputShapes_LongAxis = {
    {{{}, {{1, 1024, 1024}}}},
    {{{1, -1, 1024}, {{1, 512, 1024}, {1, 1024, 1024}}}},
};

std::vector<std::map<std::string, ov::element::Type>> config_infer_prec_f16 = {
    {{ov::hint::inference_precision.name(), ov::element::f16}}};

const auto params_LongAxis_f16 = testing::Combine(testing::Combine(testing::Values(std::vector<int>{1}),
                                                                   testing::Values(ov::test::utils::OpType::VECTOR),
                                                                   testing::ValuesIn(keepDims()),
                                                                   testing::Values(ov::test::utils::ReductionType::Mean,
                                                                                   ov::test::utils::ReductionType::Sum,
                                                                                   ov::test::utils::ReductionType::L1,
                                                                                   ov::test::utils::ReductionType::L2),
                                                                   testing::ValuesIn(inpOutPrc()),
                                                                   testing::Values(ElementType::dynamic),
                                                                   testing::Values(ElementType::dynamic),
                                                                   testing::ValuesIn(inputShapes_LongAxis)),
                                                  testing::Values(emptyCPUSpec),
                                                  testing::Values(emptyFusingSpec),
                                                  testing::ValuesIn(config_infer_prec_f16));

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_LongAxis_f16_CPU,
                         ReduceCPULayerTest,
                         params_LongAxis_f16,
                         ReduceCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace Reduce
}  // namespace test
}  // namespace ov
