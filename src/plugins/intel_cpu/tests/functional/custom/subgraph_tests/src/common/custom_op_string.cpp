// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//    -------------------------------          -------------------
//   |           Parameter           |        |     Constant      |
//    -------------------------------          -------------------
//      | string     | string     | string        | string     | string
//  --------    -----------    -----------------------    -----------
// | Result |  | Extension |  |        Extension      |  | Extension |
//  --------    -----------    -----------------------    -----------
//                  | u8            | string       | string     | u8
//              --------      -----------      --------      --------
//             | Result |    | Extension |    | Result |    | Result |
//              --------      -----------      --------      --------
//                                  | u8
//                           ---------------
//                          | Bitwise (CPU) |
//                           ---------------
//                                  | u8
//                              --------
//                             | Result |
//                              --------

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

using CustomOpStringCPUTestParams = std::tuple<ElementType, InputShape>;

class CustomOpStringString : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpStringString");

    CustomOpStringString() = default;
    CustomOpStringString(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 2, "Input count must be 2, Got: ", inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == ov::element::Type_t::string, "The input must be string.");
        OPENVINO_ASSERT(get_input_element_type(1) == ov::element::Type_t::string, "The input must be string.");

        set_output_size(2);
        set_output_type(0, ov::element::Type_t::string, get_input_partial_shape(0));
        set_output_type(1, ov::element::Type_t::string, get_input_partial_shape(1));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments: ", new_args.size(), ". 2 is expected.");
        return std::make_shared<CustomOpStringString>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        for (size_t i = 0lu; i < inputs.size(); i++) {
            OPENVINO_ASSERT(inputs[i].get_shape().size() == static_cast<size_t>(get_input_partial_shape(i).rank().get_length()),
                "Invalid input shape rank: ", inputs[i].get_shape().size());
        }
        for (size_t i = 0lu; i < outputs.size(); i++) {
            OPENVINO_ASSERT(outputs[i].get_shape().size() == static_cast<size_t>(get_output_partial_shape(i).rank().get_length()),
                "Invalid outputs shape rank: ", outputs[i].get_shape().size());
        }

        auto in_data_0 = inputs[0].data<ov::element_type_traits<ov::element::string>::value_type>();
        auto in_data_1 = inputs[1].data<ov::element_type_traits<ov::element::string>::value_type>();
        auto out_data_0 = outputs[0].data<ov::element_type_traits<ov::element::string>::value_type>();
        auto out_data_1 = outputs[1].data<ov::element_type_traits<ov::element::string>::value_type>();

        const auto el_num_0 = outputs[0].get_size();
        for (size_t i = 0lu; i < el_num_0; i++) {
            out_data_0[i] = in_data_0[i];
        }

        const auto el_num_1 = outputs[1].get_size();
        for (size_t i = 0lu; i < el_num_1; i++) {
            out_data_1[i] = in_data_1[i];
        }

        return true;
    }

    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override {
        return evaluate(output_values, input_values);
    }

    bool has_evaluate() const override { return true; }
};

class CustomOpStringU8 : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpStringU8");

    CustomOpStringU8() = default;
    CustomOpStringU8(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1, "Input count must be 1, Got: ", inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == ov::element::Type_t::string, "The input must be string.");

        set_output_size(1);
        set_output_type(0, ov::element::Type_t::u8, get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments: ", new_args.size(), ". 1 is expected.");
        return std::make_shared<CustomOpStringU8>(new_args);
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        for (size_t i = 0lu; i < inputs.size(); i++) {
            OPENVINO_ASSERT(inputs[i].get_shape().size() == static_cast<size_t>(get_input_partial_shape(i).rank().get_length()),
                "Invalid input shape rank: ", inputs[i].get_shape().size());
        }
        for (size_t i = 0lu; i < outputs.size(); i++) {
            OPENVINO_ASSERT(outputs[i].get_shape().size() == static_cast<size_t>(get_output_partial_shape(i).rank().get_length()),
                "Invalid outputs shape rank: ", outputs[i].get_shape().size());
        }

        auto in_data_0 = inputs[0].data<ov::element_type_traits<ov::element::string>::value_type>();
        auto out_data_0 = outputs[0].data<ov::element_type_traits<ov::element::u8>::value_type>();

        const auto el_num_0 = outputs[0].get_size();
        for (size_t i = 0lu; i < el_num_0; i++) {
            if (in_data_0[i].empty()) {
                out_data_0[i] = '_';
            } else {
                out_data_0[i] = in_data_0[i][0];
            }
        }

        return true;
    }

    bool has_evaluate() const override { return true; }
    bool visit_attributes(ov::AttributeVisitor& visitor) override { return true; }
};

class CustomOpStringCPUTest : public testing::WithParamInterface<CustomOpStringCPUTestParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpStringCPUTestParams>& obj) {
        ElementType in_type;
        InputShape inputShape;
        std::tie(in_type, inputShape) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << in_type;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        ElementType in_type;
        InputShape inputShape;
        std::tie(in_type, inputShape) = this->GetParam();

        init_input_shapes({inputShape});

        auto in_0 = std::make_shared<ov::op::v0::Parameter>(in_type, inputDynamicShapes[0]);
        auto in_1 = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(in_type, { 1, 3, 5 }));
        auto str_str_op = std::make_shared<CustomOpStringString>(ov::OutputVector{in_0, in_1});
        auto str_u8_op_0 = std::make_shared<CustomOpStringU8>(ov::OutputVector{str_str_op});
        auto str_u8_op_1 = std::make_shared<CustomOpStringU8>(ov::OutputVector{in_0});
        auto str_u8_op_2 = std::make_shared<CustomOpStringU8>(ov::OutputVector{in_1});
        auto btw_not_op = std::make_shared<ov::op::v13::BitwiseNot>(str_u8_op_0->output(0));

        ov::ParameterVector input_params{in_0};
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(btw_not_op->output(0)),
                                 std::make_shared<ov::op::v0::Result>(str_str_op->output(1)),
                                 std::make_shared<ov::op::v0::Result>(str_u8_op_1->output(0)),
                                 std::make_shared<ov::op::v0::Result>(str_u8_op_2->output(0)),
                                 std::make_shared<ov::op::v0::Result>(in_0->output(0))};
        function = std::make_shared<ov::Model>(results, input_params, "CustomOpStringString");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());

        auto expected_data_0 = expected[0].data<ov::element_type_traits<ov::element::u8>::value_type>();
        auto actual_data_0 = actual[0].data<ov::element_type_traits<ov::element::u8>::value_type>();
        const auto size_0 = expected[0].get_size();

        for (size_t i = 0lu; i < size_0; i++) {
            OPENVINO_ASSERT(expected_data_0[i] == actual_data_0[i], "At index ", i,
                " expected: '", expected_data_0[i], "' actual: '", actual_data_0[i], "'");
        }

        auto expected_data_1 = expected[1].data<ov::element_type_traits<ov::element::string>::value_type>();
        auto actual_data_1 = actual[1].data<ov::element_type_traits<ov::element::string>::value_type>();
        const auto size_1 = expected[1].get_size();

        for (size_t i = 0lu; i < size_1; i++) {
            OPENVINO_ASSERT(expected_data_1[i] == actual_data_1[i], "At index ", i,
                " expected: '", expected_data_1[i], "' actual: '", actual_data_1[i], "'");
        }
    }
};

TEST_P(CustomOpStringCPUTest, CompareWithRefs) {
    run();
}

const std::vector<InputShape> inputShapes = {
    {{}, {{2, 5}}},
    {{}, {{17, 9}}},
    {{-1, -1}, {{1, 3}, {5, 17}, {99, 51}}},
    {{}, {{}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         CustomOpStringCPUTest,
                         ::testing::Combine(::testing::Values(ElementType::string), ::testing::ValuesIn(inputShapes)),
                         CustomOpStringCPUTest::getTestCaseName);

} // namespace SubgraphTestsDefinitions
