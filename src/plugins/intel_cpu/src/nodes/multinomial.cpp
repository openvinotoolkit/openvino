// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include "ie_parallel.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/op/multinomial.hpp>
#include <openvino/reference/multinomial.hpp>
#include "shape_inference/custom/multinomial.hpp"


namespace ov {
namespace intel_cpu {
namespace node {

Multinomial::Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, MultinomialShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << (errorMessage);
    }

    auto multinomial_op = as_type_ptr<op::v13::Multinomial>(op);
    m_convert_type = multinomial_op->get_convert_type();
    m_with_replacement = multinomial_op->get_with_replacement();
    m_log_probs = multinomial_op->get_log_probs();
    m_global_seed = multinomial_op->get_global_seed();
    m_op_seed = multinomial_op->get_op_seed();

    m_errorPrefix = "Multinomial node with name '" + getName() + "' ";
}

bool Multinomial::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v13::Multinomial::get_type_info_static()) {
            errorMessage = "Only Multinomial operation from the opset13 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Multinomial::getSupportedDescriptors() {
    if (getParentEdges().size() != 2) {
        IE_THROW() << m_errorPrefix << "has incorrect number of input edges.";
    }
    if (getChildEdges().size() != 1) {
        IE_THROW() << m_errorPrefix <<  "has incorrect number of output edges.";
    }
}

void Multinomial::initSupportedPrimitiveDescriptors() {
    auto probs_prc = getOriginalInputPrecisionAtPort(PROBS_PORT);
    if (!one_of(probs_prc, InferenceEngine::Precision::FP64, InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::BF16)) {
        probs_prc = InferenceEngine::Precision::FP32;
    }

    auto num_samples_prc = getOriginalInputPrecisionAtPort(NUM_SAMPLES_PORT);
    if (!one_of(num_samples_prc, InferenceEngine::Precision::I64, InferenceEngine::Precision::I32)) {
        num_samples_prc = InferenceEngine::Precision::I32;
    }

    auto out_prc = getOriginalOutputPrecisionAtPort(OUTPUT_PORT);
    if (!one_of(out_prc, InferenceEngine::Precision::I64, InferenceEngine::Precision::I32)) {
        out_prc = InferenceEngine::Precision::I32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, probs_prc},
                          {LayoutType::ncsp, num_samples_prc}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

void Multinomial::prepareParams() {
    auto probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    auto num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();

    if (probs_shape.size() != 1 && probs_shape.size() != 2) {
        IE_THROW() << m_errorPrefix << "has incompatible 'probs' shape " << PartialShape(probs_shape) << ". Only 1D and 2D tensors are allowed.";
    }

    if (num_samples_shape.size() != 1) {
        IE_THROW() << m_errorPrefix << "has incompatible 'num_samples' shape " << PartialShape(num_samples_shape) << ". Only scalar and 1D single element tensors are allowed.";
    }
}

void Multinomial::execute(dnnl::stream strm) {
    const float* probs = reinterpret_cast<const float*>(getParentEdgeAt(PROBS_PORT)->getMemoryPtr()->getData());
    const int* num_samples = reinterpret_cast<const int*>(getParentEdgeAt(NUM_SAMPLES_PORT)->getMemoryPtr()->getData());
    const int* output = reinterpret_cast<const int*>(getParentEdgeAt(OUTPUT_PORT)->getMemoryPtr()->getData());

    auto probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    auto num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();
    auto output_shape = getParentEdgeAt(OUTPUT_PORT)->getMemory().getStaticDims();

    ov::reference::multinomial::multinomial(probs,
                                            probs_shape,
                                            num_samples,
                                            num_samples_shape,
                                            output,
                                            output_shape,
                                            m_with_replacement,
                                            m_log_probs,
                                            m_global_seed,
                                            m_op_seed
                                            );
}

void Multinomial::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Multinomial::isExecutable() const {
    return !isInputTensorAtPortEmpty(PROBS_PORT) && !isInputTensorAtPortEmpty(NUM_SAMPLES_PORT);
}

bool Multinomial::created() const {
    return getType() == Type::Multinomial;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov