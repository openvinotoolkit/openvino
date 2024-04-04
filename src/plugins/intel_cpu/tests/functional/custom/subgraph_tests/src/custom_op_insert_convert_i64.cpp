// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/op.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"


//       -------------
//      |  PARAMETER  |
//       -------------
//         |        |
//  -----------   -----------
// |  Ext_I64  | |  Ext_I64  |
//  -----------   -----------
//       |            |
//  ----------    ----------
// |  OP_I32  |  |  Result  |
//  ----------    ----------
//       |
//  ----------
// |  Result  |
//  ----------

using namespace ov::test;
using namespace CPUTestUtils;

namespace ov {
namespace test {
using CustomOpI64CPUTestParams = std::tuple<
    ElementType,     // Input element type
    InputShape       // Input shape
>;

class CustomOpI64 : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpI64");

    CustomOpI64() = default;

    CustomOpI64(const ov::OutputVector& args) : Op(args) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1,
                        "Input count must be 1, Got: ",
                        inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == element::i32 || get_input_element_type(0) == element::i64,
                        "The input must be i32 or i64.");
        set_output_size(2);

        auto in_shape = get_input_partial_shape(0);

        set_output_type(0, get_input_element_type(0), in_shape);
        set_output_type(1, element::i32, in_shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

        return std::make_shared<CustomOpI64>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& in = inputs[0];
        auto& out_0 = outputs[0];
        auto& out_1 = outputs[1];

        memcpy(out_0.data(), in.data(), out_0.get_byte_size());

        if (get_input_element_type(0) == element::i32) {
            memcpy(out_1.data(), in.data(), out_1.get_byte_size());
        } else if (get_input_element_type(0) == element::i64) {
            auto in_data = in.data<int64_t>();

            auto out_data_1 = out_1.data<int32_t>();
            for (size_t i = 0lu; i < in.get_size(); i++) {
                out_data_1[i] = static_cast<int32_t>(in_data[i]);
            }
        } else {
            OPENVINO_THROW("Upexpected input element type: ", get_input_element_type(0));
        }

        return true;
    }

    bool evaluate(ov::TensorVector& output_values,
                  const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override {
        return evaluate(output_values, input_values);
    }

    bool has_evaluate() const override {
        return true;
    }
};

class CustomOpConvertI64CPUTest : public testing::WithParamInterface<CustomOpI64CPUTestParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CustomOpI64CPUTestParams>& obj) {
        std::ostringstream result;

        result << "InElType=" << std::get<0>(obj.param);
        result << "_IS="      << std::get<1>(obj.param);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const auto& params = this->GetParam();
        m_in_el_type = std::get<0>(params);
        const auto& in_shape = std::get<1>(params);

        init_input_shapes({in_shape});
        ov::ParameterVector in_params;
        ov::OutputVector params_outs;
        for (auto&& shape : inputDynamicShapes) {
            auto param = std::make_shared<op::v0::Parameter>(m_in_el_type, shape);
            in_params.push_back(param);
            params_outs.push_back(param);
        }
        auto custom_op_0 = std::make_shared<CustomOpI64>(params_outs);
        auto custom_op_1 = std::make_shared<CustomOpI64>(params_outs);
        auto logical_op = std::make_shared<op::v1::LogicalNot>(custom_op_0);

        ov::ResultVector results{
                std::make_shared<op::v0::Result>(logical_op),
                std::make_shared<op::v0::Result>(custom_op_0->output(1)),
                std::make_shared<op::v0::Result>(custom_op_1->output(0)),
                std::make_shared<op::v0::Result>(custom_op_1->output(1)),
            };

        function = std::make_shared<ov::Model>(results, in_params, "customOpI64Test");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                ov::test::utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }

    ElementType m_in_el_type;
};

TEST_P(CustomOpConvertI64CPUTest, CompareWithRefs) {
    run();

    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "Convert", m_in_el_type == element::i32 ? 0 : 3);
}

const InputShape inputShapes = {
    {}, {{2, 3, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         CustomOpConvertI64CPUTest,
                         ::testing::Combine(
                                ::testing::Values(ElementType::i32, ElementType::i64),
                                ::testing::Values(inputShapes)),
                         CustomOpConvertI64CPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
