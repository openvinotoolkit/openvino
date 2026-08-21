// Copyright (C) 2018-2026 Intel Corporation
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

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

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

class CustomOpIntBuf : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpIntBuf", "gpu_opset");  // must match XML layer name

    CustomOpIntBuf() = default;

    CustomOpIntBuf(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpIntBuf>(inputs[0]);
    }

    bool has_evaluate() const override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());

        // minimal test: copy input to output
        for (size_t i = 0; i < out.get_size(); i++)
            out.data<float>()[i] = in.data<float>()[i];

        return true;
    }
};

static std::shared_ptr<ov::Model> get_simple_model_with_internal_buffer_static() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto op = std::make_shared<CustomOpIntBuf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_internal_buffer");
}

static std::shared_ptr<ov::Model> get_simple_model_with_internal_buffer_dynamic() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 2, 3, 4});
    auto op = std::make_shared<CustomOpIntBuf>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "model_with_internal_buffer");
}

TEST(CustomOpIntBuf, InternalBufferKernelStaticRuns) {
    ov::Core core;

    ov::AnyMap config = { {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH} };
    auto model = get_simple_model_with_internal_buffer_static();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(1*2*3*4);
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = float(i);

    ov::Tensor input_tensor(ov::element::f32, {1,2,3,4}, input_data.data());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i], 1e-5);
    }
}

TEST(CustomOpIntBuf, InternalBufferKernelDynamicRuns) {
    ov::Core core;

    ov::AnyMap config = { {"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH} };
    auto model = get_simple_model_with_internal_buffer_dynamic();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(1*2*3*4);
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = float(i);

    ov::Tensor input_tensor(ov::element::f32, {1,2,3,4}, input_data.data());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();

    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i], 1e-5);
    }
}

// Two input ports fed by the same producer used to build colliding pre-reorder primitive
// ids and fail to compile. Both ports declare a concrete format so a reorder is inserted
// for each.
class CustomOpTwoIn : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpTwoIn", "gpu_opset");

    CustomOpTwoIn() = default;

    CustomOpTwoIn(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) : Op({a, b}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpTwoIn>(inputs[0], inputs[1]);
    }
};

TEST(CustomOpTwoIn, SameProducerOnTwoPortsCompiles) {
    ov::Core core;

    const ov::Shape shape{1, 2, 3, 4};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    // Both ports deliberately read the same producer tensor.
    auto op = std::make_shared<CustomOpTwoIn>(param, param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_shared_producer_on_two_ports");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));

    auto infer_request = compiled_model.create_infer_request();
    std::vector<float> input_data(ov::shape_size(shape));
    for (size_t i = 0; i < input_data.size(); i++)
        input_data[i] = static_cast<float>(i);

    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    const float* out = output_tensor.data<const float>();
    for (size_t i = 0; i < input_data.size(); i++) {
        ASSERT_NEAR(out[i], input_data[i] * 2.0f, 1e-5f) << "element " << i;
    }
}

// Two ports of one custom op read the same producer but ask for different formats. Each
// port needs its own pre-reorder; the kernel copies port 1 (YXFB) and the output is
// declared YXFB, so the plugin reorders it back and the result must equal the input.
class CustomOpTwoInFmt : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpTwoInFmt", "gpu_opset");

    CustomOpTwoInFmt() = default;

    CustomOpTwoInFmt(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) : Op({a, b}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpTwoInFmt>(inputs[0], inputs[1]);
    }
};

TEST(CustomOpTwoInFmt, SameProducerOnTwoPortsWithDifferentFormats) {
    ov::Core core;

    const ov::Shape shape{1, 2, 2, 3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpTwoInFmt>(param, param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_two_formats_on_one_producer");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    ov::CompiledModel compiled_model;
    OV_ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config));
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(ov::shape_size(shape));
    for (size_t i = 0; i < input_data.size(); i++)
        input_data[i] = static_cast<float>(i);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);
    const float* out = output_tensor.data<const float>();
    for (size_t i = 0; i < input_data.size(); i++)
        ASSERT_NEAR(out[i], input_data[i], 1e-5f) << "element " << i;
}


} // namespace intel_gpu
} // namespace test
} // namespace ov
