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

using ConvBiasTestParams = std::tuple<nodeType,
                                      SizeVector,  // input shape
                                      SizeVector>; // kernel size

class ConvBiasTest : public testing::WithParamInterface<ConvBiasTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvBiasTestParams> obj) {
        nodeType conv;
        SizeVector IS, kernel;
        std::tie(conv, IS, kernel) = obj.param;

        std::ostringstream result;
        result << nodeType2str(conv) << "_";
        result << "K=" << CommonTestUtils::vec2str(IS) << "_";
        result << "K=" << CommonTestUtils::vec2str(kernel);

        return result.str();
    }

protected:
     std::string cpuNodeType;

    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        nodeType convType;
        SizeVector IS, kernel;
        std::tie(convType, IS, kernel) = this->GetParam();

        cpuNodeType = nodeType2PluginType(convType);

        auto inputParams = builder::makeParams(element::f32, {IS});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        std::shared_ptr<Node> conv;
        const auto convDims = kernel.size();
        const std::vector<size_t> strides(convDims, 1);
        const std::vector<ptrdiff_t> padBegin(convDims, 0);
        const std::vector<ptrdiff_t> padEnd(convDims, 0);
        const std::vector<size_t> dilation(convDims, 1);
        const size_t numOutChannels = 30;
        const size_t numOfGroups = 2;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        switch (convType) {
            case nodeType::convolution : {
                conv = builder::makeConvolution(paramOuts[0], element::f32, kernel, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
                break;
            }
            case nodeType::groupConvolution : {
                conv = builder::makeGroupConvolution(paramOuts[0], element::f32, kernel, strides, padBegin, padEnd, dilation, paddingType, numOutChannels,
                                                     numOfGroups);
                break;
            }
            default: {
                throw std::runtime_error("ConvBiasTest doesn't support this type of operation");
            }
        }

        SizeVector biasShape(conv->get_output_shape(0).size(), 1);
        biasShape[1] = conv->get_output_shape(0)[1];
        const auto bias = builder::makeConstant(element::f32, biasShape, std::vector<float>{}, true);
        const auto add = std::make_shared<ngraph::opset1::Add>(conv, bias);

        function = std::make_shared<ngraph::Function>(add, inputParams, "ConvBias");
    }
};

TEST_P(ConvBiasTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckNodeOfTypeCount(executableNetwork, cpuNodeType, 1);
    CheckNodeOfTypeCount(executableNetwork, "Eltwise", 0);

    auto execGraph = executableNetwork.GetExecGraphInfo().getFunction();
    ASSERT_NE(nullptr, execGraph);

    bool foundConv = false;
    for (const auto &node : execGraph->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };

        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == cpuNodeType) {
            foundConv = true;
            ASSERT_EQ(3, node->inputs().size());
            break;
        }
    }

    ASSERT_TRUE(foundConv) << "Can't find " << cpuNodeType << " node";
}

const std::vector<nodeType> convType = { nodeType::convolution, nodeType::groupConvolution };

namespace conv1d {

const std::vector<SizeVector> IS = { {2, 32, 5} };
const std::vector<SizeVector> kernel = { {3} };
const auto convBiasParams = ::testing::Combine(::testing::ValuesIn(convType),
                                               ::testing::ValuesIn(IS),
                                               ::testing::ValuesIn(kernel));

INSTANTIATE_TEST_CASE_P(smoke_ConvBiasTest_1D, ConvBiasTest, convBiasParams, ConvBiasTest::getTestCaseName);

} // namespace conv1d

namespace conv2d {

const std::vector<SizeVector> IS = { {2, 32, 5, 5} };
const std::vector<SizeVector> kernel = { {3, 3} };
const auto convBiasParams = ::testing::Combine(::testing::ValuesIn(convType),
                                               ::testing::ValuesIn(IS),
                                               ::testing::ValuesIn(kernel));

INSTANTIATE_TEST_CASE_P(smoke_ConvBiasTest_2D, ConvBiasTest, convBiasParams, ConvBiasTest::getTestCaseName);

} // namespace conv2d

namespace conv3d {

const std::vector<SizeVector> IS = { {2, 32, 5, 5, 5} };
const std::vector<SizeVector> kernel = { {3, 3, 3} };
const auto convBiasParams = ::testing::Combine(::testing::ValuesIn(convType),
                                               ::testing::ValuesIn(IS),
                                               ::testing::ValuesIn(kernel));

INSTANTIATE_TEST_CASE_P(smoke_ConvBiasTest_3D, ConvBiasTest, convBiasParams, ConvBiasTest::getTestCaseName);

} // namespace conv3d

} // namespace SubgraphTestsDefinitions
