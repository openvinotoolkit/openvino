// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <random>
#include <string>

#include "openvino/core/parallel.hpp"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Multinomial : public Node {
public:
    Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool needPrepareParams() const override;
    void prepareParams() override;

    void createPrimitive() override;

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
    bool m_const_batch = false;
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
    void execute_probs_type();

    template <typename P, typename O>
    void execute_convert_type();
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
