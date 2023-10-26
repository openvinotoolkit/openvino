// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
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
    void createPrimitive() override;

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
    ov::element::Type_t m_convert_type = ov::element::i32;
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
    bool m_probs_1d = false;
    size_t m_input_elements_count = 0;
    size_t m_batches_count = 0;
    size_t m_samples_count = 0;
    size_t m_probs_count = 0;
    std::vector<float> m_cdf;
    std::vector<float> m_max_per_batch;
    std::vector<double> m_random_samples;
    std::vector<int> m_output_vals;

    /// RandomUniform jit kernel params
    // struct ThreadParams {
    //     uint64_t work_amount = 0lu;
    //     uint64_t dst_shift = 0lu;
    //     uint64_t n_shift = 0lu;
    //     uint64_t step = 0lu;
    // };

    // uint64_t m_threads_num = 0lu;
    // std::vector<ThreadParams> m_thread_params;
    // std::shared_ptr<kernel::JitKernelBase> m_jit_random_uniform_kernel;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
