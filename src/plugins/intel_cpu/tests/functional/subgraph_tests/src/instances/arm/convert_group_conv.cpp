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

using namespace CPUTestUtils;
using namespace ov::test;
using namespace ngraph;
using namespace ngraph::helpers;

namespace CPUSubgraphTestsDefinitions {

typedef std::tuple<InputShape> groupConvLayerCPUTestParamsSet;

class GroupConvToConvTransformationCPUTest: public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerCPUTestParamsSet> obj) {
        InputShape inputShapes;
        std::tie(inputShapes) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShapes;

        return result.str();
    }

protected:
    static const size_t numOfGroups = 2;
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape inputShapes;
        std::tie(inputShapes) = this->GetParam();

        init_input_shapes({inputShapes});

        std::shared_ptr<Node> conv;
        const std::vector<size_t> kernelSize = {1};
        const std::vector<size_t> strides = {1};
        const std::vector<ptrdiff_t> padBegin = {0};
        const std::vector<ptrdiff_t> padEnd = {0};
        const std::vector<size_t> dilation = {1};
        const size_t numOutChannels = 30;
        const op::PadType paddingType = op::PadType::EXPLICIT;

        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, shape));
        }
        conv = builder::makeGroupConvolution(inputParams[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation,
                                             paddingType, numOutChannels, numOfGroups);

        ResultVector results;
        results.push_back(std::make_shared<opset5::Result>(conv));

        function = std::make_shared<ngraph::Function>(results, inputParams, "groupConvolution");
    }
};

TEST_P(GroupConvToConvTransformationCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Split", 1);
    CheckNumberOfNodesWithType(compiledModel, "Convolution", numOfGroups);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 1);
}

namespace {
std::vector<InputShape> inShapes = {
    {{}, {{ 2, 12, 7 }}},
    {
        //dynamic shape
        {-1, 12, {1, 20}},
        { //target static shapes
            { 2, 12, 7 },
            { 1, 12, 5 }
        }
    }
};
const auto groupConvTransformationParams = ::testing::Combine(::testing::ValuesIn(inShapes));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvToConvTransformationTest, GroupConvToConvTransformationCPUTest,
                         groupConvTransformationParams, GroupConvToConvTransformationCPUTest::getTestCaseName);

} // namespace
} // namespace CPUSubgraphTestsDefinitions
