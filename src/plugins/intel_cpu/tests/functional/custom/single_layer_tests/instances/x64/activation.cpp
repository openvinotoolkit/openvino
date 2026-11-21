// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/activation.hpp"

#include <cmath>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/erf.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test::utils;

namespace ov {
namespace test {
namespace Activation {
namespace {

TEST(smoke_ErfCPU, SmallInputs) {
    const std::vector<float> values{3e-12f, -3e-12f, 9e-20f, -9e-20f, 7e-8f, -7e-8f, 1e-4f, -1e-4f, 0.4f, -0.4f};
    const ov::Shape shape{values.size()};
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto erf = std::make_shared<ov::op::v0::Erf>(input);
    const auto model = std::make_shared<ov::Model>(erf->outputs(), ov::ParameterVector{input});
    auto request = PluginCache::get().core()->compile_model(model, DEVICE_CPU).create_infer_request();
    request.set_input_tensor(create_tensor(ov::element::f32, shape, values));
    request.infer();

    const auto output = request.get_output_tensor();
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data<const float>()[i], std::erf(values[i])) << "input: " << values[i];
    }
}

const std::vector<ov::element::Type>& netPrc() {
    static const std::vector<ov::element::Type> netPrc {
        ov::element::f32,
        ov::element::bf16,
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
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3Dblocked())),
    ::testing::Values(false)
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
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4Dblocked())),
    ::testing::Values(false)
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
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5Dblocked())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU_Blocked, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Activation
}  // namespace test
}  // namespace ov
