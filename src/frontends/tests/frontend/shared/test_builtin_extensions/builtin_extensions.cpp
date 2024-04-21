// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/opsets/opset8.hpp"

#ifdef ENABLE_OV_ONNX_FRONTEND
#    include "openvino/frontend/onnx/extension/conversion.hpp"
#    define ONNX_EXT                                                                                      \
        std::make_shared<ov::frontend::onnx::ConversionExtension>("NewCustomOp_3", CustomTranslatorONNX), \
            std::make_shared<ov::frontend::onnx::ConversionExtension>("Relu", ReluToSwishTranslator),
#else
#    define ONNX_EXT
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
#    include "openvino/frontend/paddle/extension/conversion.hpp"
#    define PADDLE_EXT                                                                                            \
        std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle),     \
            std::make_shared<ov::frontend::paddle::ConversionExtension>("relu", ReluToSwishTranslatorPDPD),       \
            std::make_shared<ov::frontend::paddle::ConversionExtension>("NewCustomOp_4", CustomTranslatorPaddle), \
            std::make_shared<ov::frontend::paddle::ConversionExtension>("relu6", Relu6ToReluTranslatorPaddle),
#else
#    define PADDLE_EXT
#endif

#ifdef ENABLE_OV_TF_FRONTEND
#    include "openvino/frontend/tensorflow/extension/conversion.hpp"
#    define TF_EXT                                                                                                    \
        std::make_shared<ov::frontend::tensorflow::ConversionExtension>("NewCustomOp_5", CustomTranslatorTensorflow), \
            std::make_shared<ov::frontend::tensorflow::ConversionExtension>("Relu", ReluToSwishTranslator),
#else
#    define TF_EXT
#endif

#ifdef ENABLE_OV_TF_LITE_FRONTEND
#    include "openvino/frontend/tensorflow_lite/extension/conversion.hpp"
#    define TF_LITE_EXT                                                                                   \
        std::make_shared<ov::frontend::tensorflow_lite::ConversionExtension>("NewCustomOp_6",             \
                                                                             CustomTranslatorTensorflow), \
            std::make_shared<ov::frontend::tensorflow_lite::ConversionExtension>("RELU", ReluToSwishTranslator),
#else
#    define TF_LITE_EXT
#endif

namespace {

ov::OutputVector CustomTranslatorCommon_1(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

std::map<std::string, ov::OutputVector> CustomTranslatorCommon_2(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}

#if defined(ENABLE_OV_TF_FRONTEND) || defined(ENABLE_OV_TF_LITE_FRONTEND)
ov::OutputVector CustomTranslatorTensorflow(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}
#endif

#ifdef ENABLE_OV_ONNX_FRONTEND
ov::OutputVector CustomTranslatorONNX(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}
#endif

#if defined(ENABLE_OV_TF_FRONTEND) || defined(ENABLE_OV_TF_LITE_FRONTEND) || defined(ENABLE_OV_ONNX_FRONTEND) || \
    defined(ENABLE_OV_PYTORCH_FRONTEND)
ov::OutputVector ReluToSwishTranslator(const ov::frontend::NodeContext& node) {
    return {std::make_shared<ov::opset8::Swish>(node.get_input(0))};
}
#endif

#ifdef ENABLE_OV_PADDLE_FRONTEND
std::map<std::string, ov::OutputVector> ReluToSwishTranslatorPDPD(const ov::frontend::NodeContext& node) {
    return {{"Out", {std::make_shared<ov::opset8::Swish>(node.get_input("X"))}}};
}

std::map<std::string, ov::OutputVector> CustomTranslatorPaddle(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}

std::map<std::string, ov::OutputVector> Relu6ToReluTranslatorPaddle(const ov::frontend::NodeContext& node) {
    auto relu = std::make_shared<ov::opset8::Relu>(node.get_input("X"));
    std::map<std::string, ov::OutputVector> ret;
    ret["Out"] = {relu};
    return ret;
}
#endif
}  // namespace

class CustomElu : public ov::op::Op {
public:
    OPENVINO_OP("CustomElu");

    CustomElu() = default;
    CustomElu(const ov::Output<ov::Node>& input, float alpha, float beta) : m_alpha{alpha}, m_beta{beta} {
        set_argument(0, input);
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomElu>(inputs[0], m_alpha, m_beta);
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("m_alpha", m_alpha);
        visitor.on_attribute("m_beta", m_beta);
        return true;
    }

    bool has_evaluate() const override {
        switch (get_input_element_type(0)) {
        case ov::element::f32:
            return true;
        default:
            return false;
        }
    }

    template <typename T>
    void elu(const T* input, T* output, size_t n) const {
        for (size_t i = 0; i < n; i++) {
            if (input[i] > 0)
                output[i] = m_beta * input[i];
            else
                output[i] = m_alpha * (std::exp(input[i]) - 1);
        }
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        switch (get_input_element_type(0)) {
        case ov::element::f32:
            elu(inputs[0].data<float>(), outputs[0].data<float>(), ov::shape_size(get_output_shape(0)));
            break;
        default:
            return false;
        }
        return true;
    }

private:
    float m_alpha;
    float m_beta;
};

#ifdef ENABLE_OV_PYTORCH_FRONTEND
#    include "openvino/frontend/extension/op.hpp"
#    include "openvino/frontend/pytorch/extension/conversion.hpp"
#    include "openvino/frontend/pytorch/extension/op.hpp"
#    include "openvino/op/relu.hpp"
class ReluCustom : public ov::op::v0::Relu {
public:
    OPENVINO_OP("ReluCustom");
    OPENVINO_FRAMEWORK_MAP(pytorch, "aten::relu");
};
#    define PT_EXT                                                                                       \
        std::make_shared<ov::frontend::pytorch::OpExtension<CustomElu>>(                                 \
            "aten::elu",                                                                                 \
            std::map<std::string, size_t>{{"m_alpha", 1}},                                               \
            std::map<std::string, ov::Any>{{"m_beta", 1.0f}}),                                           \
            std::make_shared<ov::frontend::pytorch::ConversionExtension>("Relu", ReluToSwishTranslator), \
            std::make_shared<ov::OpExtension<ReluCustom>>(),

#else
#    define PT_EXT
#endif

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_1", CustomTranslatorCommon_1),
     std::make_shared<ov::frontend::ConversionExtension>("NewCustomOp_2", CustomTranslatorCommon_2),
     ONNX_EXT PADDLE_EXT TF_EXT TF_LITE_EXT PT_EXT}));
