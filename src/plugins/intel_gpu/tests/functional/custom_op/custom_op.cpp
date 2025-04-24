// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

#include "base/ov_behavior_test_utils.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {

class CustomOp : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomOp", "gpu_opset");

    CustomOp() = default;

    CustomOp(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
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

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOp>(inputs[0], m_alpha, m_beta);
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

static std::shared_ptr<ov::Model> get_simple_model_with_custom_op() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto op = std::make_shared<CustomOp>(param, 1.0f, 2.0f);
    auto result = std::make_shared<ov::op::v0::Result>(op);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_custom_op");
}

TEST(CustomOp, CanReadValidCustomOpConfig) {
    ov::Core core;
    core.set_property(ov::test::utils::DEVICE_GPU, {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}});
}

TEST(CustomOp, NoRedundantReordersInserted) {
    ov::Core core;
    auto model = get_simple_model_with_custom_op();
    ov::AnyMap config = { ov::hint::inference_precision(ov::element::f32), {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);

    auto runtime_graph = compiled_model.get_runtime_model();

    auto ops = runtime_graph->get_ordered_ops();
    ASSERT_EQ(ops.size(), 3);
    ASSERT_STREQ(ops[0]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Input");
    ASSERT_STREQ(ops[1]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "CustomGPUPrimitive");
    ASSERT_STREQ(ops[2]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Result");
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
