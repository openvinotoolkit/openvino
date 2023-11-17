// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>
#include <string>

#include "ie_common.h"
#include "ie_parallel.hpp"
#include "node.h"

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

protected:
    bool needShapeInfer() const override;

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
    ov::element::Type m_probs_precision;
    ov::element::Type m_num_samples_precision;
    ov::element::Type m_output_precision;

    size_t m_probs_count = 0;
    size_t m_batches_count = 0;
    size_t m_samples_count = 0;
    size_t m_samples_probs_count = 0;
    size_t m_input_elements_count = 0;
    size_t m_output_elements_count = 0;
    size_t m_batches_samples_probs_count = 0;

    template <typename P>
    void execute_types() {
        switch (m_output_precision) {
        case ov::element::i32:
            return execute_internal<P, int32_t>();
        case ov::element::i64:
            return execute_internal<P, int64_t>();
        default:
            THROW_CPU_NODE_ERR("Multinomial CPU implementation does not support output convert type: ",
                               m_output_precision);
        }
    }

    template <typename P, typename O>
    void execute_internal() {
        const auto* probs = reinterpret_cast<const P*>(getParentEdgeAt(PROBS_PORT)->getMemoryPtr()->getData());
        auto* output = reinterpret_cast<O*>(getChildEdgeAt(OUTPUT_PORT)->getMemoryPtr()->getData());

        std::vector<P> m_cdf(m_input_elements_count);
        std::vector<P> m_max_per_batch(m_batches_count);
        std::vector<P> m_random_samples(m_output_elements_count);

        // exp & cumsum
        if (m_log_probs) {
            parallel_for(m_batches_count, [&](size_t idx) {
                auto start_idx = idx * m_probs_count;
                m_cdf[start_idx] = std::exp(probs[start_idx]);
                for (size_t prev = start_idx, curr = prev + 1; curr < (start_idx + m_probs_count); ++prev, ++curr) {
                    m_cdf[curr] = std::exp(probs[curr]) + m_cdf[prev];
                }
            });
        } else {
            parallel_for(m_batches_count, [&](size_t idx) {
                auto start_idx = idx * m_probs_count;
                m_cdf[start_idx] = probs[start_idx];
                for (size_t prev = start_idx, curr = prev + 1; curr < (start_idx + m_probs_count); ++prev, ++curr) {
                    m_cdf[curr] = probs[curr] + m_cdf[prev];
                }
            });
        }

        // TODO RandomUniform - should use RandomUniform kernel to match other frameworks' seed results
        uint64_t seed;
        if (m_global_seed == 0 && m_op_seed == 0) {
            seed = std::time(NULL);
        } else {
            std::srand(m_global_seed);
            std::srand(m_op_seed + std::rand());
            seed = std::rand();
        }

        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t idx = 0lu; idx < m_output_elements_count; ++idx) {
            m_random_samples[idx] = static_cast<P>(dist(gen));
        }

        // max & divide
        const auto min_value_of_max = std::numeric_limits<P>::min();
        parallel_for(m_batches_count, [&](size_t idx) {
            m_max_per_batch[idx] = std::max(m_cdf[(idx + 1) * m_probs_count - 1], min_value_of_max);
        });
        parallel_for(m_input_elements_count, [&](size_t idx) {
            size_t idx_max_elem = idx / m_probs_count;
            m_cdf[idx] /= m_max_per_batch[idx_max_elem];
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
        } else {
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
                        P class_probability =
                            selected_class ? m_cdf[idx_input + selected_class] - m_cdf[idx_input + selected_class - 1]
                                           : m_cdf[idx_input];
                        P divisor = 1 - class_probability;
                        for (size_t idx_prob = 0LU; idx_prob < m_probs_count; ++idx_prob) {
                            if (idx_prob >= selected_class) {
                                m_cdf[idx_input + idx_prob] -= class_probability;
                            }
                            m_cdf[idx_input + idx_prob] /= divisor;
                        }
                    }
                }
            });
        }
    }
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
