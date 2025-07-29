// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/op/constant.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {

class CustomAddOp : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomAddOp", "gpu_opset");

    CustomAddOp() = default;

    CustomAddOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", m_alpha);
        visitor.on_attribute("beta", m_beta);
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
        return std::make_shared<CustomAddOp>(new_args[0], m_alpha, m_beta);
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());
        for (size_t i = 0; i < out.get_size(); i++) {
            out.data<float>()[i] = in.data<float>()[i] * m_alpha + m_beta;
        }
        return true;
    }
};

static std::shared_ptr<ov::Model> get_simple_model_with_custom_add_op(float alpha, float beta, ov::PartialShape inp_shape) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inp_shape);
    auto op = std::make_shared<CustomAddOp>(input, alpha, beta);
    auto result = std::make_shared<ov::op::v0::Result>(op);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "model_with_custom_op_dynamic");
}

TEST(CustomOpDynamic, CanReadValidCustomOpConfig) {
    ov::Core core;
    core.set_property(ov::test::utils::DEVICE_GPU, {{"CONFIG_FILE", TEST_CUSTOM_OP_DYNAMIC_CONFIG_PATH}});
}

TEST(smoke_CustomOpDynamic, Accuracy) {
    ov::Core core;
    float alpha = 1.0, beta = 0.1;
    const size_t dim1 = 1;
    auto model = get_simple_model_with_custom_add_op(alpha, beta, ov::PartialShape{-1, dim1, -1});

    ov::AnyMap config = {ov::hint::inference_precision(ov::element::f32), {"CONFIG_FILE", TEST_CUSTOM_OP_DYNAMIC_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);

    auto runtime_graph = compiled_model.get_runtime_model();
    auto ops = runtime_graph->get_ordered_ops();

    bool found_custom_op = false;
    for (auto op : ops) {
        if (op->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>() == "CustomGPUPrimitive") {
            found_custom_op = true;
            break;
        }
    }
    ASSERT_TRUE(found_custom_op);

    auto inp_arr_1 = std::vector<float>{0.2, 0.4};
    auto inp_arr_2 = std::vector<float>{0.2, 0.4, 0.3, 0.5, 0.7, 0.9};
    auto inputs = std::vector<ov::Tensor>{ov::Tensor({ov::element::f32}, ov::Shape{1, dim1, 2}, inp_arr_1.data()),
                                          ov::Tensor({ov::element::f32}, ov::Shape{2, dim1, 3}, inp_arr_2.data())};
    auto ireq = compiled_model.create_infer_request();
    for (auto input : inputs) {
        ireq.set_input_tensor(0, input);
        ireq.infer();
        auto output = ireq.get_output_tensor(0);
        std::vector<float> actual(output.data<float>(), output.data<float>() + output.get_size());

        ASSERT_EQ(output.get_element_type(), element::f32);

        float* inp_data = input.data<float>();
        for (size_t i = 0; i < output.get_size(); i++) {
            ASSERT_FLOAT_EQ(actual[i], inp_data[i] * alpha + beta);
        }
    }
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
