// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_maxpool_activ.hpp"

namespace ov {
namespace test {
std::string ConvPoolActivTest::getTestCaseName(testing::TestParamInfo<fusingSpecificParams> obj) {
    fusingSpecificParams fusingParams = obj.param;

    std::ostringstream result;
    result << "ConvPoolActivTest";
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    return result.str();
}

void ConvPoolActivTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    fusingSpecificParams fusingParams = this->GetParam();
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    ov::ParameterVector inputParams{
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 40, 40})};

    std::shared_ptr<Node> conv;
    {
        const std::vector<size_t> kernelSize = {3, 3};
        const std::vector<size_t> strides = {2, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const std::vector<size_t> dilation = {1, 1};
        const size_t numOutChannels = 16;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        conv = ov::test::utils::make_convolution(inputParams[0],
                                                 element::f32,
                                                 kernelSize,
                                                 strides,
                                                 padBegin,
                                                 padEnd,
                                                 dilation,
                                                 paddingType,
                                                 numOutChannels);
     }
     std::shared_ptr<Node> pooling;
     {
         const std::vector<size_t> kernelSize = {3, 3};
         const std::vector<size_t> strides = {1, 1};
         const std::vector<size_t> padBegin = {0, 0};
         const std::vector<size_t> padEnd = {0, 0};
         const op::PadType paddingType = op::PadType::EXPLICIT;
         ov::op::RoundingType roundingType = ov::op::RoundingType::CEIL;
         pooling = std::make_shared<ov::op::v1::MaxPool>(conv,
                                                         strides,
                                                         padBegin,
                                                         padEnd,
                                                         kernelSize,
                                                         roundingType,
                                                         paddingType);
     }

#if defined(OPENVINO_ARCH_ARM)
    selectedType = makeSelectedTypeStr("ref_any", element::f32);
#elif defined(OPENVINO_ARCH_ARM64)
    selectedType = makeSelectedTypeStr("gemm_acl", element::f32);
#else
    selectedType = makeSelectedTypeStr(getPrimitiveType(), element::f32);
#endif
    function = makeNgraphFunction(element::f32, inputParams, pooling, "ConvPoolActiv");
}

bool ConvPoolActivTest::primTypeCheck(std::string primType) const {
#if defined(OPENVINO_ARCH_ARM)
    return primType == makeSelectedTypeStr(std::string("ref_any"), element::f32);
#elif defined(OPENVINO_ARCH_ARM64)
    return primType == makeSelectedTypeStr(std::string("gemm_acl"), element::f32);
#else
    auto isaType = getISA(true);
    if (isaType == "")
        return primType == "ref";
    else
        return primType == makeSelectedTypeStr(std::string("jit_") + isaType, element::f32) ||
               primType == makeSelectedTypeStr(std::string("brgconv_") + isaType, element::f32);
#endif
}

TEST_P(ConvPoolActivTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

}  // namespace test
}  // namespace ov