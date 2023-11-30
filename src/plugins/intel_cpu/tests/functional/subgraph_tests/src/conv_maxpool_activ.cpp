// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/convolution.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class ConvPoolActivTest : public testing::WithParamInterface<fusingSpecificParams>,
                          public CpuTestWithFusing,
                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fusingSpecificParams> obj) {
        fusingSpecificParams fusingParams = obj.param;

        std::ostringstream result;
        result << "ConvPoolActivTest";
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
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


        function = makeNgraphFunction(element::f32, inputParams, pooling, "ConvPoolActiv");
    }

    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(true);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == makeSelectedTypeStr(std::string("jit_") + isaType, element::f32) ||
                   primType == makeSelectedTypeStr(std::string("brgconv_") + isaType, element::f32);
    }
};

TEST_P(ConvPoolActivTest, CompareWithRefs) {
    selectedType = makeSelectedTypeStr(getPrimitiveType(), element::f32);
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvPoolActivTest_FP16 : public ConvPoolActivTest {
    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(true);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == selectedType;
    }
};

TEST_P(ConvPoolActivTest_FP16, CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    if (ov::with_cpu_x86_avx512_core_amx_fp16()) {
        selectedType = makeSelectedTypeStr("brgconv_avx512_amx", element::f16);
    } else {
        selectedType = makeSelectedTypeStr("brgconv_avx512", element::f16);
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {

const std::vector<fusingSpecificParams> fusingParamsSet{emptyFusingSpec, fusingRelu, fusingSwish, fusingSigmoid};

const std::vector<fusingSpecificParams> fusingParamsSet_FP16 {
        emptyFusingSpec
        // fusingRelu,
        // fusingSwish,
        // fusingSigmoid
};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ConvPoolActivTest,
                         ::testing::ValuesIn(fusingParamsSet),
                         ConvPoolActivTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ConvPoolActivTest_FP16,
                         ::testing::ValuesIn(fusingParamsSet_FP16),
                          ConvPoolActivTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
