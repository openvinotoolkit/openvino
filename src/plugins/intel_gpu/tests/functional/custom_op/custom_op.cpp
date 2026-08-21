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

// format="ANY" on an output must not change the axis order the op or the graph sees.
// The shape is non-square in Y and X, WorkSizes dispatches over X and Y separately, and
// the kernel derives its output from its dispatch coordinates: with a product-form
// WorkSizes and a flat copy kernel a transposed layout cancels out and the test cannot
// fail.
class CustomOpAnyFmt : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmt", "gpu_opset");

    CustomOpAnyFmt() = default;

    explicit CustomOpAnyFmt(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmt>(inputs[0]);
    }
};

TEST(CustomOpAnyFmt, OutputFormatAnyPreservesAxisOrder) {
    ov::Core core;

    constexpr size_t kY = 3;
    constexpr size_t kX = 4;
    const ov::Shape shape{1, 1, kY, kX};

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpAnyFmt>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_format_output");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(ov::shape_size(shape), 0.0f);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);

    const float* out = output_tensor.data<const float>();
    for (size_t y = 0; y < kY; y++) {
        for (size_t x = 0; x < kX; x++) {
            ASSERT_NEAR(out[y * kX + x], static_cast<float>(y * 1000 + x), 1e-5f)
                << "at y=" << y << " x=" << x;
        }
    }
}

// Same defect on a rank-3 output, which resolves through get_default_format(3) rather
// than (4).
class CustomOpAnyFmt3D : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmt3D", "gpu_opset");

    CustomOpAnyFmt3D() = default;

    explicit CustomOpAnyFmt3D(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmt3D>(inputs[0]);
    }
};

TEST(CustomOpAnyFmt3D, OutputFormatAnyPreservesAxisOrder3D) {
    ov::Core core;

    constexpr size_t kB = 2;
    constexpr size_t kF = 3;
    constexpr size_t kY = 4;
    const ov::Shape shape{kB, kF, kY};

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpAnyFmt3D>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_format_3d_output");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(ov::shape_size(shape), 0.0f);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);

    const float* out = output_tensor.data<const float>();
    for (size_t b = 0; b < kB; b++) {
        for (size_t f = 0; f < kF; f++) {
            for (size_t y = 0; y < kY; y++) {
                ASSERT_NEAR(out[(b * kF + f) * kY + y],
                            static_cast<float>(b * 100 + f * 10 + y), 1e-5f)
                    << "at b=" << b << " f=" << f << " y=" << y;
            }
        }
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


// format="ANY" on an output means "inherit the first input's format", resolved in
// custom_gpu_primitive_inst::calc_output_layout. With input port 0 declared
// YXFB, the kernel writes YXFB-ordered data, so the plugin must treat the output as YXFB
// and reorder it to BFYX for the Result. If ANY is instead resolved to the default
// format, the YXFB bytes are read as BFYX and the result is permuted.
class CustomOpAnyFmtYxfb : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmtYxfb", "gpu_opset");

    CustomOpAnyFmtYxfb() = default;

    explicit CustomOpAnyFmtYxfb(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmtYxfb>(inputs[0]);
    }
};

TEST(CustomOpAnyFmtYxfb, OutputFormatAnyInheritsFirstInputFormat) {
    ov::Core core;

    const ov::Shape shape{1, 2, 2, 3};
    const size_t n = ov::shape_size(shape);

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape(shape));
    auto op = std::make_shared<CustomOpAnyFmtYxfb>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_output_and_yxfb_input");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<float> input_data(n);
    for (size_t i = 0; i < n; i++)
        input_data[i] = static_cast<float>(i);
    ov::Tensor input_tensor(ov::element::f32, shape, input_data.data());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    ASSERT_EQ(output_tensor.get_shape(), shape);

    const float* out = output_tensor.data<const float>();

    for (size_t i = 0; i < n; i++)
        ASSERT_NEAR(out[i], static_cast<float>(i + 1), 1e-5f) << "at flat index " << i;
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

// The ANY sentinel survives into the dynamic path, where the format is resolved per shape
// in custom_gpu_primitive_inst::calc_output_layouts rather than at graph build time.
class CustomOpAnyFmtDyn : public ov::op::Op {
public:
    OPENVINO_OP("CustomOpAnyFmtDyn", "gpu_opset");

    CustomOpAnyFmtDyn() = default;

    explicit CustomOpAnyFmtDyn(const ov::Output<ov::Node>& input) : Op({input}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomOpAnyFmtDyn>(inputs[0]);
    }
};

TEST(CustomOpAnyFmtDyn, OutputFormatAnyOnDynamicShape) {
    ov::Core core;

    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    auto op = std::make_shared<CustomOpAnyFmtDyn>(param);
    auto result = std::make_shared<ov::op::v0::Result>(op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param},
                                             "model_with_any_format_output_dynamic");

    ov::AnyMap config = {{"CONFIG_FILE", TEST_CUSTOM_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, config);
    auto infer_request = compiled_model.create_infer_request();

    // run twice at different shapes so the layout is re-resolved between inferences
    for (const ov::Shape& shape : {ov::Shape{1, 2, 3, 4}, ov::Shape{1, 2, 5, 2}}) {
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
            ASSERT_NEAR(out[i], input_data[i] + 1.0f, 1e-5f)
                << "element " << i << " at shape " << shape;
    }
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
