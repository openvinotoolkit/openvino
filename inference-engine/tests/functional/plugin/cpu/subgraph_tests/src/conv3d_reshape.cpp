// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using Conv3dReshapeTestParams = std::tuple<nodeType,
                                           size_t>;

class Conv3dReshapeTest : public testing::WithParamInterface<Conv3dReshapeTestParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<Conv3dReshapeTestParams> obj) {
        nodeType conv;
        size_t numOut;
        std::tie(conv, numOut) = obj.param;

        std::ostringstream result;
        result << nodeType2str(conv) << "_";
        result << "NUM_OUTPUTS=" << numOut;

        return result.str();
    }

protected:
     std::string cpuNodeType;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        nodeType convType;
        size_t numOut;
        std::tie(convType, numOut) = this->GetParam();

        cpuNodeType = nodeType2PluginType(convType);

        auto inputParams = builder::makeParams(element::f32, {Shape{1, 1024, 64}});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        std::shared_ptr<Node> conv;
        const std::vector<size_t> kernelSize = {1};
        const std::vector<size_t> strides = {1};
        const std::vector<ptrdiff_t> padBegin = {0};
        const std::vector<ptrdiff_t> padEnd = {0};
        const std::vector<size_t> dilation = {1};
        const size_t numOutChannels = 30;
        const size_t numOfGroups = 2;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        switch (convType) {
            case nodeType::convolution : {
                conv = builder::makeConvolution(paramOuts[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
                break;
            }
            case nodeType::groupConvolution : {
                conv = builder::makeGroupConvolution(paramOuts[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels,
                                                     numOfGroups);
                break;
            }
            default: {
                throw std::runtime_error("Conv3dReshapeTest doesn't support this type of operation");
            }
        }

        ResultVector results;
        for (int i = 0; i < numOut; i++) {
            auto mockNode = std::make_shared<opset5::Multiply>(conv->output(0), opset5::Constant::create(element::f32, Shape{1}, {1}));
            results.push_back(std::make_shared<opset5::Result>(mockNode));
        }

        function = std::make_shared<ngraph::Function>(results, inputParams, "Conv3dReshape");
    }
};

TEST_P(Conv3dReshapeTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

namespace {

const std::vector<nodeType> convType = { nodeType::convolution, nodeType::groupConvolution };
const std::vector<size_t> numOut = { 1, 2, 5 };
const auto conv3dReshapeParams = ::testing::Combine(::testing::ValuesIn(convType),
                                                    ::testing::ValuesIn(numOut));

INSTANTIATE_TEST_CASE_P(smoke_Conv3dReshapeTest, Conv3dReshapeTest, conv3dReshapeParams, Conv3dReshapeTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
