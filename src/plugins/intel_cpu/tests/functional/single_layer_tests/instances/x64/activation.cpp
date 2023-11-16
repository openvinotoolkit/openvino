// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/activation.hpp"
#include "shared_test_classes/single_layer/activation.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {
namespace Activation {
namespace {

const std::vector<Precision>& netPrc() {
    static const std::vector<Precision> netPrc {
        Precision::FP32,
        Precision::BF16,
    };

    return netPrc;
}

const std::map<ActivationTypes, std::vector<std::vector<float>>>& activationTypesBlocked() {
    static const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesBlocked {
        {Mish,        {{}}},
        {SoftSign,    {{}}}
    };

    return activationTypesBlocked;
}

const std::vector<CPUSpecificParams>& cpuParams3Dblocked() {
    static const std::vector<CPUSpecificParams> cpuParams3Dblocked {
        CPUSpecificParams({nCw16c}, {nCw16c}, {}, {}),
    };

    return cpuParams3Dblocked;
}

const auto blockedCases3D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic3D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesBlocked())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(Precision::FP32),
    ::testing::Values(Precision::FP32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3Dblocked()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Eltwise_CPU_Blocked, ActivationLayerCPUTest, blockedCases3D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (2D) ============= */
const std::vector<CPUSpecificParams>& cpuParams4Dblocked() {
    static const std::vector<CPUSpecificParams> cpuParams4Dblocked {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
    };

    return cpuParams4Dblocked;
}

const auto basicCases4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(Precision::FP32),
    ::testing::Values(Precision::FP32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4Dblocked()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU_Blocked, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (3D) ============= */
const std::vector<CPUSpecificParams>& cpuParams5Dblocked() {
    static const std::vector<CPUSpecificParams> cpuParams5Dblocked {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
    };

    return cpuParams5Dblocked;
}

const auto basicCases5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(Precision::FP32),
    ::testing::Values(Precision::FP32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5Dblocked()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU_Blocked, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

} // namespace
} // namespace Activation
} // namespace CPULayerTestsDefinitions
