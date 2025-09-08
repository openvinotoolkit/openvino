// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/matmul.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace MatMul {
/* ============= MatMul ============= */
namespace matmul {

static const std::vector<CPUSpecificParams>& filterSpecificParamsFC() {
    static const std::vector<CPUSpecificParams> specificParams = {CPUSpecificParams{{}, {}, {"acl"}, "acl"}};
    return specificParams;
}

std::vector<fusingSpecificParams> fusingParamsSet2D_smoke {
    emptyFusingSpec,
    fusingBias,
    fusingRelu,
    fusingTanh
};

const auto testParams2D_smoke =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke()),
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(utils::InputLayerType::CONSTANT),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(emptyAdditionalConfig())),
                       ::testing::Values(MatMulNodeType::FullyConnected),
                       ::testing::ValuesIn(fusingParamsSet2D_smoke),
                       ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerCPUTest, testParams2D_smoke, MatMulLayerCPUTest::getTestCaseName);


std::vector<fusingSpecificParams> fusingParamsSet2D_smoke_f16 {
        emptyFusingSpec,
        fusingBias,
        fusingRelu
};
const auto testParams2D_smoke_f16 = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(IS2D_smoke()),
                       ::testing::Values(ElementType::f16),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(utils::InputLayerType::CONSTANT),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap({ov::hint::inference_precision(ov::element::f16)}))),
    ::testing::Values(MatMulNodeType::FullyConnected),
    ::testing::ValuesIn(fusingParamsSet2D_smoke_f16),
    ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_f16, MatMulLayerCPUTest, testParams2D_smoke_f16, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet3D_smoke {
    emptyFusingSpec,
    fusingBias,
    fusingRelu,
    fusingTanh
};
const auto fullyConnectedParams3D_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_smoke()),
                                                             ::testing::Values(ElementType::f32),
                                                             ::testing::Values(ElementType::dynamic),
                                                             ::testing::Values(ElementType::dynamic),
                                                             ::testing::Values(utils::InputLayerType::CONSTANT),
                                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                             ::testing::Values(emptyAdditionalConfig()));
std::vector<fusingSpecificParams> fusingParamsSet3D_smoke_f16 {
        emptyFusingSpec,
        fusingBias,
        fusingRelu
};
const auto fullyConnectedParams3D_smoke_f16 =
    ::testing::Combine(::testing::ValuesIn(IS3D_smoke()),
                       ::testing::Values(ElementType::f16),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(utils::InputLayerType::CONSTANT),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap({ov::hint::inference_precision(ov::element::f16)})));
const auto testParams3D_smoke = ::testing::Combine(fullyConnectedParams3D_smoke,
                                                   ::testing::Values(MatMulNodeType::FullyConnected),
                                                   ::testing::ValuesIn(fusingParamsSet3D_smoke),
                                                   ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
const auto testParams3D_smoke_f16 = ::testing::Combine(fullyConnectedParams3D_smoke_f16,
                                                   ::testing::Values(MatMulNodeType::FullyConnected),
                                                   ::testing::ValuesIn(fusingParamsSet3D_smoke_f16),
                                                   ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerCPUTest, testParams3D_smoke, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_f16, MatMulLayerCPUTest, testParams3D_smoke_f16, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS = {
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},
};

std::vector<fusingSpecificParams> fusingParamsSet4D_smoke {
        emptyFusingSpec,
        fusingRelu,
        fusingTanh
};

const auto testParams4D_smoke =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS),
                                          ::testing::Values(ElementType::f32),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(utils::InputLayerType::CONSTANT),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(emptyAdditionalConfig())),
                       ::testing::Values(MatMulNodeType::FullyConnected),
                       ::testing::ValuesIn(fusingParamsSet4D_smoke),
                       ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
INSTANTIATE_TEST_SUITE_P(smoke_FC_4D, MatMulLayerCPUTest, testParams4D_smoke, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet4D_smoke_f16 {
        emptyFusingSpec,
        fusingRelu
};

const auto testParams4D_smoke_f16 = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(IS),
                       ::testing::Values(ElementType::f16),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(ElementType::dynamic),
                       ::testing::Values(utils::InputLayerType::CONSTANT),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(ov::AnyMap({ov::hint::inference_precision(ov::element::f16)}))),
    ::testing::Values(MatMulNodeType::FullyConnected),
    ::testing::ValuesIn(fusingParamsSet4D_smoke_f16),
    ::testing::ValuesIn(filterCPUInfo(filterSpecificParamsFC())));
INSTANTIATE_TEST_SUITE_P(smoke_FC_4D_f16, MatMulLayerCPUTest, testParams4D_smoke_f16, MatMulLayerCPUTest::getTestCaseName);

}  // namespace matmul
}  // namespace MatMul
}  // namespace test
}  // namespace ov