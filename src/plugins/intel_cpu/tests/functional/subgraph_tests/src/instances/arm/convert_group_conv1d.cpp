// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ov_models/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;
using namespace ngraph;
using namespace ngraph::helpers;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<nodeType,
                   InputShape> conv1dConvertCPUTestParamsSet;

class Conv1dConvertTransformationCPUTest: public testing::WithParamInterface<conv1dConvertCPUTestParamsSet>,
                                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<conv1dConvertCPUTestParamsSet> obj) {
        InputShape inputShapes;
        nodeType convType;
        std::tie(convType, inputShapes) = obj.param;

        std::ostringstream result;
        result << nodeType2str(convType) << "_";
        result << "IS=" << inputShapes;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape inputShapes;
        nodeType convType;
        std::tie(convType, inputShapes) = this->GetParam();

        init_input_shapes({inputShapes});

        std::shared_ptr<Node> conv;
        const std::vector<size_t> kernelSize = {1};
        const std::vector<size_t> strides = {1};
        const std::vector<ptrdiff_t> padBegin = {0};
        const std::vector<ptrdiff_t> padEnd = {0};
        const std::vector<size_t> dilation = {1};
        const size_t numOutChannels = 30;
        const size_t numOfGroups = 2;
        const op::PadType paddingType = op::PadType::EXPLICIT;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, shape));
        }
        switch (convType) {
            case nodeType::convolution : {
                conv = builder::makeConvolution(inputParams[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation,
                                                paddingType, numOutChannels);
                break;
            }
            case nodeType::groupConvolution : {
                conv = builder::makeGroupConvolution(inputParams[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation,
                                                     paddingType, numOutChannels, numOfGroups);
                break;
            }
            default: {
                throw std::runtime_error("Conv1dConvertTransformationCPUTest doesn't support this type of operation");
            }
        }

        ResultVector results;
        results.push_back(std::make_shared<opset5::Result>(conv));

        function = std::make_shared<ngraph::Function>(results, inputParams, "convolution");
    }
};

TEST_P(Conv1dConvertTransformationCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Reshape", 2);
}

namespace {
const std::vector<nodeType> convType = { nodeType::convolution, nodeType::groupConvolution };
std::vector<InputShape> inputShapes1d = {
        {{}, {{ 2, 64, 7 }}},
        {{}, {{ 1, 32, 7 }}},
        {
            //dynamic shape
            { -1, 64, {1, 20} },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 9 }
            }
        },
        {
            //dynamic shape
            { -1, 32, {1, 20} },
            { //target static shapes
                { 2, 32, 7 },
                { 1, 32, 9 }
            }
        },
        {
            //dynamic shape
            { {1, 20}, 64, -1 },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 5 }
            }
        }
};

const auto groupConvTransformationParams = ::testing::Combine(::testing::ValuesIn(convType),
                                                              ::testing::ValuesIn(inputShapes1d));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvToConvTransformationTest, Conv1dConvertTransformationCPUTest,
                         groupConvTransformationParams, Conv1dConvertTransformationCPUTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
