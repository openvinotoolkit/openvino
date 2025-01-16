// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include "openvino/op/multinomial.hpp"
#include <openvino/op/constant.hpp>
#include <openvino/core/type.hpp>
#include "utils/bfloat16.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

Multinomial::Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(NUM_SAMPLES_PORT))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto multinomial_op = as_type_ptr<op::v13::Multinomial>(op);
    m_with_replacement = multinomial_op->get_with_replacement();
    m_global_seed = multinomial_op->get_global_seed();
    m_log_probs = multinomial_op->get_log_probs();
    m_op_seed = multinomial_op->get_op_seed();

    m_num_samples_precision = ov::element::i32;
    m_output_precision = multinomial_op->get_convert_type();

    constant = ConstantType::StrictNoConst;

    m_const_batch = op->get_input_partial_shape(PROBS_PORT)[0].is_static();
    m_const_inputs[PROBS_PORT] = is_type<op::v0::Constant>(op->get_input_node_ptr(PROBS_PORT));
    m_const_inputs[NUM_SAMPLES_PORT] = is_type<op::v0::Constant>(op->get_input_node_ptr(NUM_SAMPLES_PORT));
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
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Multinomial::initSupportedPrimitiveDescriptors() {
    m_probs_precision = getOriginalInputPrecisionAtPort(PROBS_PORT);
    if (!one_of(m_probs_precision, ov::element::f32, ov::element::f16, ov::element::bf16)) {
        m_probs_precision = ov::element::f32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, m_probs_precision, m_const_inputs[PROBS_PORT]},
                          {LayoutType::ncsp, m_num_samples_precision, m_const_inputs[NUM_SAMPLES_PORT]}},
                         {{LayoutType::ncsp, m_output_precision}},
                         ref_any);
}

bool Multinomial::needShapeInfer() const {
    return !(m_const_inputs[NUM_SAMPLES_PORT] && m_const_batch);
}

bool Multinomial::needPrepareParams() const {
    return true;
}

void Multinomial::createPrimitive() {
    if (!m_const_inputs[NUM_SAMPLES_PORT]) {
        CPU_NODE_ASSERT(isDynamicNode(), "is static while the samples input is a variable");
        return; // avoid reading non initialized data from the NUM_SAMPLES_PORT input
    }
    Node::createPrimitive();
}

void Multinomial::prepareParams() {
    const auto& probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    const auto& num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();

    if (probs_shape.size() != 2) {
        THROW_CPU_NODE_ERR("has incompatible 'probs' shape ",
                           PartialShape(probs_shape),
                           ". Only 2D tensors are allowed.");
    }

    if (num_samples_shape.size() != 1) {
        THROW_CPU_NODE_ERR("has incompatible 'num_samples' shape ",
                           PartialShape(num_samples_shape),
                           ". Only scalar and 1D single element tensors are allowed.");
    }

    if (m_num_samples_precision == ov::element::i32) {
        m_samples_count =
            getSrcDataAtPortAs<const int32_t>(NUM_SAMPLES_PORT)[0];
    } else {
        m_samples_count =
            getSrcDataAtPortAs<const int64_t>(NUM_SAMPLES_PORT)[0];
    }

    m_batches_count = probs_shape[0];
    m_probs_count = probs_shape[1];
    m_samples_probs_count = m_samples_count * m_probs_count;
    m_input_elements_count = m_batches_count * m_probs_count;
    m_output_elements_count = m_batches_count * m_samples_count;
    m_batches_samples_probs_count = m_output_elements_count * m_probs_count;
}

bool Multinomial::isExecutable() const {
    return !isInputTensorAtPortEmpty(PROBS_PORT) && !isInputTensorAtPortEmpty(NUM_SAMPLES_PORT);
}

bool Multinomial::created() const {
    return getType() == Type::Multinomial;
}

void Multinomial::execute(dnnl::stream strm) {
    switch (m_probs_precision) {
    case ov::element::f32:
        return execute_probs_type<float>();
    case ov::element::f16:
        return execute_probs_type<float16>();
    case ov::element::bf16:
        return execute_probs_type<bfloat16_t>();
    default:
        THROW_CPU_NODE_ERR("Multinomial CPU implementation does not support probs element type: ", m_probs_precision);
    }
}

void Multinomial::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename P>
void Multinomial::execute_probs_type() {
    switch (m_output_precision) {
    case ov::element::i32:
        return execute_convert_type<P, int32_t>();
    default:
        THROW_CPU_NODE_ERR("Multinomial CPU implementation does not support output convert type: ", m_output_precision);
    }
}

template <typename P, typename O>
void Multinomial::execute_convert_type() {
    const auto* probs = getSrcDataAtPortAs<const P>(PROBS_PORT);
    auto* output = getDstDataAtPortAs<O>(OUTPUT_PORT);

    std::vector<P> m_cdf(m_input_elements_count);
    std::vector<P> m_max_per_batch(m_batches_count);
    std::vector<P> m_random_samples(m_output_elements_count);

    // exp & cumsum
    if (m_log_probs) {
        parallel_for(m_batches_count, [&](size_t idx) {
            const auto start_idx = idx * m_probs_count;
            m_cdf[start_idx] = std::exp(probs[start_idx]);
            for (size_t prev = start_idx, curr = prev + 1; curr < (start_idx + m_probs_count); ++prev, ++curr) {
                m_cdf[curr] = std::exp(probs[curr]) + m_cdf[prev];
            }
        });
    } else {
        parallel_for(m_batches_count, [&](size_t idx_batch) {
            const auto start_idx = idx_batch * m_probs_count;
            const auto* probs_start_idx = probs + start_idx;
            std::partial_sum(probs_start_idx, probs_start_idx + m_probs_count, m_cdf.begin() + start_idx);
        });
    }

    // TODO RandomUniform - should use RandomUniform kernel to match other frameworks' seed results
    std::mt19937 gen;
    if (m_global_seed == 0 && m_op_seed == 0) {
        gen.seed(std::time(NULL));
    } else {
        std::seed_seq seed{m_global_seed, m_op_seed};
        gen.seed(seed);
    }

    const auto gen_max = static_cast<float>(gen.max());
    std::generate(m_random_samples.begin(), m_random_samples.end(), [&]() {
        return static_cast<P>(static_cast<float>(gen()) / gen_max);
    });

    // max & divide
    const auto min_value_of_max = std::numeric_limits<P>::min();
    parallel_for(m_batches_count, [&](size_t idx) {
        m_max_per_batch[idx] = std::max(m_cdf[(idx + 1) * m_probs_count - 1], min_value_of_max);
    });

    parallel_for(m_input_elements_count, [&](size_t idx) {
        size_t idx_max_elem = idx / m_probs_count;
        m_cdf[idx] = m_cdf[idx] / m_max_per_batch[idx_max_elem];
    });

    if (m_with_replacement) {
        parallel_for(m_batches_samples_probs_count, [&](size_t idx) {
            size_t idx_batch = idx / m_samples_probs_count;
            size_t idx_num_samples_probs = idx % m_samples_probs_count;
            size_t idx_prob = idx_num_samples_probs % m_probs_count;
            size_t idx_sample = idx_num_samples_probs / m_probs_count;

            size_t idx_input = idx_batch * m_probs_count + idx_prob;
            size_t idx_output = idx_batch * m_samples_count + idx_sample;
            if (m_random_samples[idx_output] <= m_cdf[idx_input] &&
                (!idx_prob || m_random_samples[idx_output] > m_cdf[idx_input - 1])) {
                output[idx_output] = static_cast<O>(idx_prob);
            }
        });
    } else {  // without replacement - adjust cdf after each sample drawn from batch, sequentially
        parallel_for(m_batches_count, [&](size_t idx_batch) {
            for (size_t idx_sample = 0LU; idx_sample < m_samples_count; ++idx_sample) {
                size_t idx_input = idx_batch * m_probs_count;
                size_t idx_output = idx_batch * m_samples_count + idx_sample;

                bool class_selected = false;
                size_t selected_class = m_probs_count;
                P sample_value = m_random_samples[idx_output];
                for (size_t idx_prob = 0LU; idx_prob < m_probs_count; ++idx_prob) {
                    if (sample_value <= m_cdf[idx_input + idx_prob]) {
                        output[idx_output] = static_cast<O>(idx_prob);
                        selected_class = idx_prob;
                        class_selected = true;
                        break;
                    }
                }

                if (class_selected) {
                    P class_probability;
                    if (selected_class) {
                        class_probability = m_cdf[idx_input + selected_class] - m_cdf[idx_input + selected_class - 1];
                    } else {
                        class_probability = m_cdf[idx_input];
                    }
                    P divisor = 1 - class_probability;
                    for (size_t idx_prob = 0LU; idx_prob < m_probs_count; ++idx_prob) {
                        if (idx_prob >= selected_class) {
                            m_cdf[idx_input + idx_prob] = m_cdf[idx_input + idx_prob] - class_probability;
                        }
                        m_cdf[idx_input + idx_prob] = m_cdf[idx_input + idx_prob] / divisor;
                    }
                }
            }
        });
    }
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
