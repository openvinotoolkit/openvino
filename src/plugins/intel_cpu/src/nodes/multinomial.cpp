// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include <openvino/op/multinomial.hpp>
#include <openvino/reference/multinomial.hpp>

#include "ie_ngraph_utils.hpp"
#include "ie_parallel.hpp"
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
        IE_THROW() << m_errorPrefix << "has incorrect number of output edges.";
    }
}

void Multinomial::initSupportedPrimitiveDescriptors() {
    auto probs_prc = getOriginalInputPrecisionAtPort(PROBS_PORT);
    if (!one_of(probs_prc,
                InferenceEngine::Precision::FP64,
                InferenceEngine::Precision::FP32,
                InferenceEngine::Precision::FP16,
                InferenceEngine::Precision::BF16)) {
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

    addSupportedPrimDesc({{LayoutType::nspc, probs_prc}, {LayoutType::nspc, num_samples_prc}},
                         {{LayoutType::nspc, out_prc}},
                         ref_any);
}

void Multinomial::prepareParams() {
    auto probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    auto num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();

    if (probs_shape.size() != 1 && probs_shape.size() != 2) {
        IE_THROW() << m_errorPrefix << "has incompatible 'probs' shape " << PartialShape(probs_shape)
                   << ". Only 1D and 2D tensors are allowed.";
    }

    if (num_samples_shape.size() != 1) {
        IE_THROW() << m_errorPrefix << "has incompatible 'num_samples' shape " << PartialShape(num_samples_shape)
                   << ". Only scalar and 1D single element tensors are allowed.";
    }

    m_probs_1d = probs_shape.size() == 1;

    m_input_elements_count = std::accumulate(probs_shape.front(), probs_shape.back(), 1, std::multiplies<size_t>());
    m_batches_count = probs_shape.size() == 2 ? probs_shape[0] : 1;
    m_probs_count = probs_shape.size() == 2 ? probs_shape[1] : probs_shape[0];

    m_cdf.reserve(m_input_elements_count);
    m_input_vals.reserve(m_input_elements_count);
    m_random_samples.reserve(m_input_elements_count);
    m_max_per_batch.reserve(probs_shape.size() == 1 ? 1 : probs_shape[1]);
}

void Multinomial::execute(dnnl::stream strm) {
    const auto* probs = reinterpret_cast<const float*>(getParentEdgeAt(PROBS_PORT)->getMemoryPtr()->getData());
    const auto num_samples =
        reinterpret_cast<const int*>(getParentEdgeAt(NUM_SAMPLES_PORT)->getMemoryPtr()->getData())[0];
    auto* output = reinterpret_cast<int*>(getChildEdgeAt(OUTPUT_PORT)->getMemoryPtr()->getData());

    // exp & cumsum
    if (m_log_probs) {
        parallel_for(m_input_elements_count, [&](size_t idx) {
            m_input_vals[idx] = std::exp(probs[idx]);
        });
        m_cdf[0] = m_input_vals[0];
        for (size_t idx = 1; idx < m_input_elements_count; ++idx) {
            m_cdf[idx] = m_input_vals[idx] + m_cdf[idx - 1];
        }
    } else {
        m_cdf[0] = probs[0];
        for (size_t idx = 1; idx < m_input_elements_count; ++idx) {
            m_cdf[idx] = probs[idx] + m_cdf[idx - 1];
        }
    }

    // TODO, MOCK RandomUniform
    parallel_for(m_input_elements_count, [&](size_t idx) {
        m_random_samples[idx] = static_cast<double>(idx % m_probs_count) / m_probs_count;
    });

    // max (slice) & divide
    if (m_probs_1d) {
        m_max_per_batch[0] = m_cdf[m_input_elements_count - 1];
        parallel_for(m_input_elements_count, [&](size_t idx) {
            m_cdf[idx] /= m_max_per_batch[0];
        });
    } else {
        parallel_for(m_batches_count, [&](size_t idx) {
            m_max_per_batch[idx] = m_cdf[idx * m_batches_count - 1];
        });
        parallel_for(m_input_elements_count, [&](size_t idx) {
            size_t idx_max_elem = idx / m_probs_count;
            m_cdf[idx] /= m_max_per_batch[idx_max_elem];
        });
    }

    if (m_with_replacement) {
        parallel_for(m_input_elements_count, [&](size_t idx) {
            size_t idx_output = idx / m_probs_count * num_samples;
            size_t idx_class = idx % m_probs_count;
            if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                output[idx_output] = static_cast<int>(idx_class);
            }
        });
    } else {
        parallel_for(m_input_elements_count, [&](size_t idx) {
            size_t idx_batch = idx / m_probs_count;
            size_t idx_class = idx % m_probs_count;
            size_t idx_output = idx_batch * num_samples;

            float class_probability = 0.0f;
            float divisor = 0.0f;

            size_t selected_class = 0;
            if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                output[idx_output] = static_cast<int>(idx_class);
                selected_class = idx_class;
                class_probability = m_input_vals[idx];
                divisor = 1 - class_probability;
            }

            parallel_for(m_probs_count, [&](size_t probs_idx) {
                size_t idx_start = m_probs_count * idx_batch;
                if (probs_idx >= selected_class) {
                    m_cdf[idx_start + probs_idx] -= class_probability;
                }
                m_cdf[idx_start + probs_idx] /= divisor;
            });
        });
    }
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov