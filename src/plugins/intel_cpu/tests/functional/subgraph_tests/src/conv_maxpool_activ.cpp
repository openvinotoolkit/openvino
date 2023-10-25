// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ConvPoolActivTestParams = fusingSpecificParams;

class ConvPoolActivTest : public testing::WithParamInterface<ConvPoolActivTestParams>, public CpuTestWithFusing,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvPoolActivTestParams> obj) {
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

        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 40, 40})};
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {2, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 16;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = builder::makeConvolution(paramOuts[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
        }
        std::shared_ptr<Node> pooling;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<size_t> padBegin = {0, 0};
            const std::vector<size_t> padEnd = {0, 0};
            const op::PadType paddingType = op::PadType::EXPLICIT;
            ngraph::helpers::PoolingTypes poolType = ngraph::helpers::PoolingTypes::MAX;
            ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::CEIL;
            pooling = builder::makePooling(conv, strides, padBegin, padEnd, kernelSize, roundingType, paddingType, false, poolType);
        }

        selectedType = makeSelectedTypeStr(getPrimitiveType(), element::f32);

        function = makeNgraphFunction(element::f32, inputParams, pooling, "ConvPoolActiv");
    }

    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(true);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == makeSelectedTypeStr(std::string("jit_") + isaType, element::f32)
                || primType == makeSelectedTypeStr(std::string("brgconv_") + isaType, element::f32);
    }
};

TEST_P(ConvPoolActivTest, CompareWithRefs) {
    Run();
    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {

const std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingRelu,
        fusingSwish,
        fusingSigmoid
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ConvPoolActivTest, ::testing::ValuesIn(fusingParamsSet), ConvPoolActivTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
