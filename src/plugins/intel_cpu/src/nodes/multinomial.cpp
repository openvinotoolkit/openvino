// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include <openvino/op/multinomial.hpp>

#include "ie_ngraph_utils.hpp"
#include "ie_parallel.hpp"
#include "kernels/x64/random_uniform.hpp"
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
    m_with_replacement = multinomial_op->get_with_replacement();
    m_convert_type = multinomial_op->get_convert_type();
    m_global_seed = multinomial_op->get_global_seed();
    m_log_probs = multinomial_op->get_log_probs();
    m_op_seed = multinomial_op->get_op_seed();

    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }
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
    auto probs_prc = getOriginalInputPrecisionAtPort(PROBS_PORT);
    if (!one_of(probs_prc,
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

    addSupportedPrimDesc({{LayoutType::ncsp, probs_prc, m_const_inputs[PROBS_PORT]},
                          {LayoutType::ncsp, num_samples_prc, m_const_inputs[NUM_SAMPLES_PORT]}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

std::string Multinomial::getPrimitiveDescriptorType() const {
    std::string str_type;
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    if (type == impl_desc_type::unknown)
        str_type += "unknown_";
    if ((type & impl_desc_type::jit) == impl_desc_type::jit)
        str_type += "jit_";
    if ((type & impl_desc_type::ref) == impl_desc_type::ref)
        str_type += "ref_";
    if ((type & impl_desc_type::avx512) == impl_desc_type::avx512)
        str_type += "avx512_";
    if ((type & impl_desc_type::avx2) == impl_desc_type::avx2)
        str_type += "avx2_";
    if ((type & impl_desc_type::sse42) == impl_desc_type::sse42)
        str_type += "sse42_";
    if ((type & impl_desc_type::any) == impl_desc_type::any)
        str_type += "any_";

    if (str_type.empty())
        str_type += "undef_";

    if (selectedPrimitiveDesc) {
        str_type += std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().name());
    } else {
        str_type.pop_back();
    }

    return str_type;
}

void Multinomial::createPrimitive() {
    // #if defined(OPENVINO_ARCH_X86_64)
    //     kernel::RandomUniformCompileParams jcp;

    //     jcp.out_data_type = ov::element::f32;

    //     m_jit_random_uniform_kernel = kernel::JitKernel<kernel::RandomUniformCompileParams,
    //     kernel::RandomUniformCallArgs>::createInstance<kernel::RandomUniform>(jcp);

    //     if (m_jit_random_uniform_kernel) {
    //         if (auto selected_pd = getSelectedPrimitiveDescriptor()) {
    //             using namespace dnnl::impl::cpu;
    //             if (m_jit_random_uniform_kernel->getIsa() == x64::avx512_core) {
    //                 selected_pd->setImplementationType(jit_avx512);
    //             } else if (m_jit_random_uniform_kernel->getIsa() == x64::avx2) {
    //                 selected_pd->setImplementationType(jit_avx2);
    //             } else if (m_jit_random_uniform_kernel->getIsa() == x64::sse41) {
    //                 selected_pd->setImplementationType(jit_sse42);
    //             }
    //         }
    //     }
    // #endif // OPENVINO_ARCH_X86_64

    if (m_const_inputs[PROBS_PORT] && m_const_inputs[NUM_SAMPLES_PORT]) {
        Node::createPrimitive();
    }
}

bool Multinomial::needPrepareParams() const {
    if (m_output_shape != getChildEdgeAt(0)->getMemoryPtr()->getShape().getStaticDims()) {
        return true;
    }
    return false;
}

void Multinomial::prepareParams() {
    const auto probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    const auto num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();
    const auto num_samples = reinterpret_cast<const int*>(
        getParentEdgeAt(NUM_SAMPLES_PORT)->getMemoryPtr()->getData())[0];  // reuse in execute?

    if (probs_shape.size() != 1 && probs_shape.size() != 2) {
        THROW_CPU_NODE_ERR("has incompatible 'probs' shape ",
                           PartialShape(probs_shape),
                           ". Only 1D and 2D tensors are allowed.");
    }

    if (num_samples_shape.size() != 1) {
        THROW_CPU_NODE_ERR("has incompatible 'num_samples' shape ",
                           PartialShape(num_samples_shape),
                           ". Only scalar and 1D single element tensors are allowed.");
    }

    m_probs_1d = probs_shape.size() == 1;

    m_input_elements_count = std::accumulate(probs_shape.begin(), probs_shape.end(), 1, std::multiplies<size_t>());
    m_batches_count = probs_shape.size() == 2 ? probs_shape[0] : 1;
    m_probs_count = probs_shape.size() == 2 ? probs_shape[1] : probs_shape[0];

    m_cdf.reserve(m_input_elements_count);
    m_input_vals.reserve(m_input_elements_count);
    m_random_samples.reserve(m_input_elements_count);
    m_max_per_batch.reserve(probs_shape.size() == 1 ? 1 : probs_shape[1]);

    if (!m_probs_1d) {
        m_output_shape.push_back(m_batches_count);
    }
    m_output_shape.push_back(num_samples);
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

    // TODO RandomUniform - should use RandomUniform kernel to match other frameworks' seed results
    std::srand(m_op_seed);
    for (size_t idx = 0lu; idx < m_input_elements_count; ++idx) {
        m_random_samples[idx] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    };

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
            float divisor = 1.0f;

            size_t selected_class = 0;
            if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                output[idx_output] = static_cast<int>(idx_class);
                selected_class = idx_class;
                class_probability = m_input_vals[idx];
                divisor = 1 - class_probability;
            }

            size_t idx_start = m_probs_count * idx_batch;
            for (size_t probs_idx = 0lu; probs_idx < m_probs_count; ++probs_idx) {
                if (probs_idx >= selected_class) {
                    m_cdf[idx_start + probs_idx] -= class_probability;
                }
                m_cdf[idx_start + probs_idx] /= divisor;
            };
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
