// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include "ie_parallel.hpp"
#include <node.h>

#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

class Multinomial : public Node {
public:
    Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    std::string getPrimitiveDescriptorType() const override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool needPrepareParams() const override;
    void prepareParams() override;

    bool isExecutable() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    /// Multinomial params
    bool m_with_replacement = false;
    bool m_log_probs = false;
    uint64_t m_global_seed = 0;
    uint64_t m_op_seed = 0;

    /// Shape inference
    static constexpr size_t PROBS_PORT = 0lu;
    static constexpr size_t NUM_SAMPLES_PORT = 1lu;
    static constexpr size_t OUTPUT_PORT = 0lu;
    bool m_const_inputs[2] = {false, false};
    VectorDims m_output_shape = {};

    /// General algorithm variables
    InferenceEngine::Precision m_probs_precision;
    InferenceEngine::Precision m_num_samples_precision;
    InferenceEngine::Precision m_output_precision;

    size_t m_input_elements_count = 0;
    size_t m_batches_count = 0;
    size_t m_samples_count = 0;
    size_t m_probs_count = 0;

    template <typename P>
    void execute_types() {
        switch(m_output_precision) {
            case InferenceEngine::Precision::I32:
                return execute_internal<P, int32_t>();
            default:
                THROW_CPU_NODE_ERR("Multinomial CPU implementation does not support output convert type: ", m_output_precision);
        }
    }

    template <typename P, typename O>
    void execute_internal() {
        const auto* probs = reinterpret_cast<const P*>(getParentEdgeAt(PROBS_PORT)->getMemoryPtr()->getData());
        auto* output = reinterpret_cast<O*>(getChildEdgeAt(OUTPUT_PORT)->getMemoryPtr()->getData());

        std::vector<P> m_cdf(m_input_elements_count);
        std::vector<P> m_max_per_batch(m_batches_count);
        std::vector<P> m_random_samples(m_input_elements_count);

        std::cout << "probs" << "\n";
        for(size_t q = 0; q < m_input_elements_count; q++) {
            std::cout << probs[q] << " ";
        } std::cout << "\n";
        std::cout << "num_samples" << "\n";
        std::cout << m_samples_count << " ";
        std::cout << "\n";

        // exp & cumsum
        if (m_log_probs) {
            parallel_for(m_batches_count, [&](size_t idx) {
                auto start_idx = idx * m_probs_count;
                m_cdf[start_idx] = std::exp(probs[start_idx]);
                for (size_t i = 1; i < m_probs_count; ++i) {
                    m_cdf[start_idx + i] = std::exp(probs[start_idx + i]) + m_cdf[start_idx + i - 1];
                }
            });
        } else {
            parallel_for(m_batches_count, [&](size_t idx) {
                auto start_idx = idx * m_probs_count;
                m_cdf[start_idx] = probs[start_idx];
                for (size_t i = 1; i < m_probs_count; ++i) {
                    m_cdf[start_idx + i] = probs[start_idx + i] + m_cdf[start_idx + i - 1];
                }
            });
        }
        std::cout << "cdf" << "\n";
        for(size_t q = 0; q < m_input_elements_count; q++) {
            std::cout << m_cdf[q] << " ";
        } std::cout << "\n";

        // TODO RandomUniform - should use RandomUniform kernel to match other frameworks' seed results
        std::srand(m_op_seed);
        for (size_t idx = 0lu; idx < m_input_elements_count; ++idx) {
            m_random_samples[idx] = static_cast<P>(std::rand()) / static_cast<P>(RAND_MAX);
        };
        std::cout << "rand" << "\n";
        for(size_t q = 0; q < m_input_elements_count; q++) {
            std::cout << m_random_samples[q] << " ";
        } std::cout << "\n";

        // max (slice) & divide
        const auto min_max_value = std::numeric_limits<P>::min();
        parallel_for(m_batches_count, [&](size_t idx) {
            m_max_per_batch[idx] = std::max(m_cdf[(idx + 1) * m_probs_count - 1], min_max_value);
        });
        parallel_for(m_input_elements_count, [&](size_t idx) {
            size_t idx_max_elem = idx / m_probs_count;
            m_cdf[idx] /= m_max_per_batch[idx_max_elem];
        });

        std::cout << "max" << "\n";
        for(size_t q = 0; q < m_batches_count; q++) {
            std::cout << m_max_per_batch[q] << " ";
        } std::cout << "\n";
        std::cout << "cdf_2" << "\n";
        for(size_t q = 0; q < m_input_elements_count; q++) {
            std::cout << m_cdf[q] << " ";
        } std::cout << "\n";

        if (m_with_replacement) {
            parallel_for(m_input_elements_count, [&](size_t idx) {
                size_t idx_output = idx / m_probs_count * m_samples_count;
                size_t idx_class = idx % m_probs_count;
                if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                    output[idx_output] = static_cast<O>(idx_class);
                }
            });
        } else if (m_log_probs) {
            parallel_for(m_input_elements_count, [&](size_t idx) {
                size_t idx_batch = idx / m_probs_count;
                size_t idx_class = idx % m_probs_count;
                size_t idx_output = idx_batch * m_samples_count;

                P class_probability = 0.0f;
                P divisor = 1.0f;

                size_t selected_class = 0;
                if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                    output[idx_output] = static_cast<O>(idx_class);
                    selected_class = idx_class;
                    class_probability = std::exp(probs[idx]); // only difference between log_probs vs no log_probs
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
        } else {
            parallel_for(m_input_elements_count, [&](size_t idx) {
                size_t idx_batch = idx / m_probs_count;
                size_t idx_class = idx % m_probs_count;
                size_t idx_output = idx_batch * m_samples_count;

                P class_probability = 0.0f;
                P divisor = 1.0f;

                size_t selected_class = 0;
                if (m_random_samples[idx] <= m_cdf[idx] && (!idx_class || m_random_samples[idx] > m_cdf[idx - 1])) {
                    output[idx_output] = static_cast<O>(idx_class);
                    selected_class = idx_class;
                    class_probability = probs[idx];
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
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
