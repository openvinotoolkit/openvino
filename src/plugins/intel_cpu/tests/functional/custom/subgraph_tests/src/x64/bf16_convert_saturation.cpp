// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {
/*
  This test aims to cover Eltwise node BF16 output precision conversion logic in "saturation" mode. In this test, we
  have a select node with condition input of boolean type and then/else inputs of f32 type(as constant node with bf16
  overflow data). The select node is followed by a convolution node to ensoure that it is converted to bf16 precision.
*/
using selectParams = std::tuple<InputShape,  // Condition shapes
                                ElementType  // Then/Else precision
                                >;
class BF16ConvertSaturation : public testing::WithParamInterface<selectParams>,
                              virtual public SubgraphBaseTest,
                              public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<selectParams> obj) {
        InputShape shapes;
        ElementType precision;
        std::tie(shapes, precision) = obj.param;

        std::ostringstream result;
        result << "Condition_prc_" << ElementType::boolean << "_Then_Else_prc_" << precision << "_";
        result << "IS=(" << shapes.first << ")_TS=(";
        for (const auto& item : shapes.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "PluginConf_inference_precision=bf16";

        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape shapes;
        ElementType precision;
        std::tie(shapes, precision) = this->GetParam();
        init_input_shapes({shapes});
        std::tie(inFmts, outFmts, priority, selectedType) = emptyCPUSpec;
        selectedType = makeSelectedTypeStr(getPrimitiveType(), ov::element::i8);
        ov::element::TypeVector types{ov::element::boolean, precision, precision};
        ov::ParameterVector parameters;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes[0]);
        parameters.push_back(param);

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -3.40282e+38;
        in_data.range = 1;
        in_data.resolution = 1;
        auto thenTensor = ov::test::utils::create_and_fill_tensor(precision, ov::Shape{1}, in_data);

        in_data.start_from = 1;
        in_data.range = 10;
        in_data.resolution = 2;
        auto elseTensor = ov::test::utils::create_and_fill_tensor(precision, ov::Shape{2, 1, 32, 32}, in_data);

        auto select = std::make_shared<ov::op::v1::Select>(parameters[0],
                                                           std::make_shared<ov::op::v0::Constant>(thenTensor),
                                                           std::make_shared<ov::op::v0::Constant>(elseTensor),
                                                           ov::op::AutoBroadcastType::NUMPY);

        auto conv_filter_shape = ov::Shape{1, 1, 3, 3};
        auto conv_filter = ov::op::v0::Constant::create(ElementType::f32, conv_filter_shape, {1});
        auto strides = ov::Strides{1, 1};
        auto pads_begin = ov::CoordinateDiff{0, 0};
        auto pads_end = ov::CoordinateDiff{0, 0};
        auto dilations = ov::Strides{1, 1};
        auto conv =
            std::make_shared<ov::op::v1::Convolution>(select, conv_filter, strides, pads_begin, pads_end, dilations);

        function = makeNgraphFunction(ElementType::f32, parameters, conv, "Eltwise");
        configuration.insert({ov::hint::inference_precision(ov::element::bf16)});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 3;
        in_data.resolution = 2;
        auto condTensor = ov::test::utils::create_and_fill_tensor(modelInputs[0].get_element_type(),
                                                                  targetInputStaticShapes[0],
                                                                  in_data);

        inputs.insert({modelInputs[0].get_node_shared_ptr(), condTensor});
    }
};

TEST_P(BF16ConvertSaturation, CompareWithRefs) {
    run();
}

const std::vector<InputShape> inShapes = {
    // Condition
    {{-1, -1, -1, -1}, {{2, 1, 32, 32}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BF16ConvertSaturationTest,
                         BF16ConvertSaturation,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::Values(ElementType::f32)),
                         BF16ConvertSaturation::getTestCaseName);

}  // namespace test
}  // namespace ov