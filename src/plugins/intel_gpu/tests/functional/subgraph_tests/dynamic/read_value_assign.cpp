// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ngraph;
using namespace ov::test;
using namespace InferenceEngine;

namespace GPULayerTestsDefinitions {

using ReadValueAssignParams = std::tuple<
    InputShape,  // input shapes
    ElementType  // input precision
>;

class ReadValueAssignGPUTest : virtual public SubgraphBaseTest, public testing::WithParamInterface<ReadValueAssignParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        InputShape input_shapes;
        ElementType input_precision;
        std::tie(input_shapes, input_precision) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : input_shapes.second) {
            result << ov::test::utils::partialShape2str({shape}) << "_";
        }
        result << "Precision=" << input_precision;
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape input_shapes;
        ElementType input_precision;
        std::tie(input_shapes, input_precision) = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;

        init_input_shapes({input_shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(input_precision, shape));
        }
        const VariableInfo variable_info { inputDynamicShapes[0], input_precision, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(params.at(0), variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params.at(0));
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        function = std::make_shared<ov::Model>(ResultVector { res }, SinkVector { assign }, params);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto data_tensor = ov::Tensor{funcInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto data = data_tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
        auto len = ov::shape_size(targetInputStaticShapes[0]);
        for (size_t i = 0; i < len; i++) {
            data[i] = static_cast<int>(i);
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), data_tensor});
    }
};

TEST_P(ReadValueAssignGPUTest, CompareWithRefs) {
   SKIP_IF_CURRENT_TEST_IS_DISABLED()
   run();
}

namespace {
const std::vector<InputShape> input_shapes_dyn = {
    {{-1, -1, -1, -1}, {{7, 4, 20, 20}, {19, 4, 20, 20}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Static, ReadValueAssignGPUTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation({{7, 4, 20, 20}})),
                                            ::testing::Values(ov::element::i32)),
                         ReadValueAssignGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Dynamic, ReadValueAssignGPUTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_dyn),
                                            ::testing::Values(ov::element::i32)),
                         ReadValueAssignGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
