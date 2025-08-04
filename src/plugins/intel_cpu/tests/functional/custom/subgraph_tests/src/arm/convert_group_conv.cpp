// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/group_convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<InputShape> groupConvLayerCPUTestParamsSet;

class GroupConvToConvTransformationCPUTest: public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvLayerCPUTestParamsSet>& obj) {
        const auto& [inputShapes] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShapes;

        return result.str();
    }

protected:
    static const size_t numOfGroups = 2;
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [inputShapes] = this->GetParam();
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
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        conv = utils::make_group_convolution(inputParams[0],
                                             element::f32,
                                             kernelSize,
                                             strides,
                                             padBegin,
                                             padEnd,
                                             dilation,
                                             paddingType,
                                             numOutChannels,
                                             numOfGroups);

        ResultVector results;
        results.push_back(std::make_shared<ov::op::v0::Result>(conv));

        function = std::make_shared<ov::Model>(results, inputParams, "groupConvolution");
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

}  // namespace
}  // namespace test
}  // namespace ov
