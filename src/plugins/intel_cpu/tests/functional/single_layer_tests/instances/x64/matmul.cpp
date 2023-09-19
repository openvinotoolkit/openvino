// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/matmul.hpp"
#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace MatMul {

std::vector<fusingSpecificParams> fusingParamsSet2D_nightly {
        fusingRelu,
#ifndef OV_CPU_WITH_MLAS
        fusingScaleShift, //covered by MLAS
#endif
        fusingPReluPerTensor,
        fusingFakeQuantizePerChannelRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet2D_smoke {
// The following three patterns are covered by MLAS test
#ifndef OV_CPU_WITH_MLAS
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel,
#endif
        fusingFakeQuantizePerTensorRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet2DBF16 {
        emptyFusingSpec,
        fusingBias,
        fusingRelu,
        fusingPReluPerTensor,
};

const auto testParams2DBF16_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke()),
                                                                    ::testing::ValuesIn(netPRCs),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                    ::testing::ValuesIn(additionalConfig)),
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet2DBF16),
                                                 ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerCPUTest, testParams2D_smoke, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_BF16, MatMulLayerCPUTest, testParams2DBF16_smoke, MatMulLayerCPUTest::getTestCaseName);

const auto testParams2D_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke()),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_smoke),
                                             ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));



const auto testParams2D_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly()),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values((emptyAdditionalConfig()))),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

const auto fullyConnectedParams2D_Brgemm_Amx_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_Amx_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testParams2D_Brgemm_Amx_smoke = ::testing::Combine(fullyConnectedParams2D_Brgemm_Amx_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_Brgemm_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_Brgemm_Amx, MatMulLayerCPUTest, testParams2D_Brgemm_Amx_smoke, MatMulLayerCPUTest::getTestCaseName);


const auto fullyConnectedParams2D_Brgemm_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testParams2D_Brgemm_nightly = ::testing::Combine(fullyConnectedParams2D_Brgemm_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_Brgemm(true)));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_Brgemm, MatMulLayerCPUTest, testParams2D_Brgemm_nightly, MatMulLayerCPUTest::getTestCaseName);

const auto fullyConnectedParams2D_Brgemm_Amx_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testParams2D_Brgemm_Amx_nightly = ::testing::Combine(fullyConnectedParams2D_Brgemm_Amx_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_Brgemm_Amx, MatMulLayerCPUTest, testParams2D_Brgemm_Amx_nightly, MatMulLayerCPUTest::getTestCaseName);

const auto testParams2DBF16_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly()),
                                                                    ::testing::ValuesIn(netPRCs),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                    ::testing::ValuesIn(additionalConfig)),
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet2DBF16),
                                                 ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D, MatMulLayerCPUTest, testParams2D_nightly, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_BF16, MatMulLayerCPUTest, testParams2DBF16_nightly, MatMulLayerCPUTest::getTestCaseName);

const auto fullyConnectedParams3D_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_smoke()),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::Values(emptyAdditionalConfig()));

const auto testParams3D_smoke = ::testing::Combine(fullyConnectedParams3D_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_smoke),
                                             ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerCPUTest, testParams3D_smoke, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet3D_nightly {
        fusingFakeQuantizePerTensorRelu,
};

const auto testParams3D_nightly = ::testing::Combine(fullyConnectedParams3D_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_nightly),
                                             ::testing::ValuesIn(filterCPUInfo(filterSpecificParams())));

INSTANTIATE_TEST_SUITE_P(nightly_FC_3D, MatMulLayerCPUTest, testParams3D_nightly, MatMulLayerCPUTest::getTestCaseName);

#ifdef OV_CPU_WITH_MLAS
std::vector<CPUSpecificParams> filterSpecificParams_MLAS() {
    // replace with mlas primitive type
    std::vector<CPUSpecificParams> specificParams;
    specificParams.push_back(CPUSpecificParams{{}, {}, {"gemm_mlas"}, "gemm_mlas"});
    return specificParams;
}

std::vector<fusingSpecificParams> fusingParamsSet3D_MLAS_smoke {
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel
};

const auto testParams3D_MLAS_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS3D_smoke()),
                                                        ::testing::Values(ElementType::f32),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                        ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_MLAS_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_MLAS, MatMulLayerCPUTest, testParams3D_MLAS_smoke, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet2D_MLAS_nightly {
        fusingScaleShift
};
const auto testParams2D_MLAS_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly()),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_MLAS_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_MLAS, MatMulLayerCPUTest, testParams2D_MLAS_nightly, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> fusingParamsSet2D_MLAS_smoke {
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel
};

const auto testParams2D_MLAS_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke()),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig())),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_MLAS_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_MLAS, MatMulLayerCPUTest, testParams2D_MLAS_smoke, MatMulLayerCPUTest::getTestCaseName);
#endif

} // namespace MatMul
} // namespace CPULayerTestsDefinitions