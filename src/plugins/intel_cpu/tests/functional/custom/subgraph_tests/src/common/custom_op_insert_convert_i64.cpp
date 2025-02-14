// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/op.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

//         ---------------------                                                ---------------------
//        |      PARAMETER      |                                              |      PARAMETER      |
//         ---------------------                                                ---------------------
//            |              |                                                    |               |
//    -------------        ----------                                     -------------        ----------
//   |   EXT_I64   |      |  CPU_OP  |                                   |   EXT_I64   |      |  CPU_OP  |
//    -------------        ----------             -------\                -------------        ----------
//      |i64     |i32           |                         \                 |i64     |i32           |
//  --------   ---------   -------------                  /        --------------   ---------   -------------
// | CPU_OP | | RES_I32 | |   EXT_I64   |         -------/        | CVT I64->I32 | | RES_I32 | |   EXT_I64   |
//  --------   ---------   -------------                           --------------   ---------   -------------
//      |i64                 |i64    |i32                                 |i32                    |i64    |i32
//  ---------          ---------   ---------                          --------              ---------   ---------
// | RES_I64 |        | RES_I64 | | RES_I32 |                        | CPU_OP |            | RES_I64 | | RES_I32 |
//  ---------          ---------   ---------                          --------              ---------   ---------
//                                                                        |i32
//                                                                 --------------
//                                                                | CVT I32->I64 |
//                                                                 --------------
//                                                                        |i64
//                                                                    ---------
//                                                                   | RES_I64 |
//                                                                    ---------

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

    CustomOpI64(const Output<Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        const auto& inputs_count = input_values().size();
        OPENVINO_ASSERT(inputs_count == 1, "Input count must be 1, Got: ", inputs_count);
        OPENVINO_ASSERT(get_input_element_type(0) == element::i32, "The input must be i32.");
        set_output_size(2);

        auto in_shape = get_input_partial_shape(0);

        set_output_type(0, element::i64, in_shape);
        set_output_type(1, element::i32, in_shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

        return std::make_shared<CustomOpI64>(new_args[0]);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& in = inputs[0];
        auto& out0 = outputs[0];
        auto& out1 = outputs[1];

        auto inData = in.data<int32_t>();

        auto outData0 = out0.data<int64_t>();
        for (size_t i = 0lu; i < in.get_size(); i++) {
            outData0[i] = static_cast<int64_t>(inData[i]);
        }

        memcpy(out1.data(), inData, out1.get_byte_size());

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

        result << "InElType=" << std::get<0>(obj.param) << "_";
        result << "IS="       << std::get<1>(obj.param);

        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        targetDevice = test::utils::DEVICE_CPU;

        const auto& params = this->GetParam();
        const auto& in_el_type = std::get<0>(params);
        const auto& in_shape = std::get<1>(params);

        init_input_shapes({in_shape});
        auto param = std::make_shared<op::v0::Parameter>(in_el_type, inputDynamicShapes[0]);

        auto custom_op_0 = std::make_shared<CustomOpI64>(param);
        auto i32_op_0 = std::make_shared<op::v1::LogicalNot>(custom_op_0->output(0));
        auto i32_op_1 = std::make_shared<op::v1::LogicalNot>(param);
        auto custom_op_1 = std::make_shared<CustomOpI64>(i32_op_1);

        OutputVector results{
                i32_op_0->output(0),
                custom_op_0->output(1),
                custom_op_1->output(0),
                custom_op_1->output(1)
            };

        function = std::make_shared<ov::Model>(results, ParameterVector{param}, "customOpI64Test");
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
};

TEST_P(CustomOpConvertI64CPUTest, CompareWithRefs) {
    run();

    size_t cvt_i64_i32_num = 0lu;
    size_t cvt_i32_i64_num = 0lu;
    size_t res_i64_num = 0lu;
    size_t res_i32_num = 0lu;

    for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
        auto rt_info = node->get_rt_info();
        auto it = rt_info.find(exec_model_info::LAYER_TYPE);
        ASSERT_NE(rt_info.end(), it);

        if (it->second.as<std::string>() == "Convert") {
            if (node->get_output_element_type(0) == element::i64) {
                cvt_i32_i64_num++;
            } else if (node->get_output_element_type(0) == element::i32) {
                cvt_i64_i32_num++;
            } else {
                FAIL() << "Unexpected convertion type: " << node->get_output_element_type(0);
            }
        }
        if (it->second.as<std::string>() == "Output") {
            if (node->get_output_element_type(0) == element::i64) {
                res_i64_num++;
            } else if (node->get_output_element_type(0) == element::i32) {
                res_i32_num++;
            } else {
                FAIL() << "Unexpected Result type: " << node->get_output_element_type(0);
            }
        }
    }

    ASSERT_EQ(cvt_i32_i64_num, 1lu) << "Unexpected number of the Convert i32->i64 nodes.";
    ASSERT_EQ(cvt_i64_i32_num, 1lu) << "Unexpected number of the Convert i64->i32 nodes.";
    ASSERT_EQ(res_i64_num, 2lu) << "Unexpected number of the Result nodes with type i64.";
    ASSERT_EQ(res_i32_num, 2lu) << "Unexpected number of the Result nodes with type i32.";
}

const InputShape inputShapes = {
    {}, {{2, 3, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         CustomOpConvertI64CPUTest,
                         ::testing::Combine(
                                ::testing::Values(ElementType::i32),
                                ::testing::Values(inputShapes)),
                         CustomOpConvertI64CPUTest::getTestCaseName);
}  // namespace test
}  // namespace ov
