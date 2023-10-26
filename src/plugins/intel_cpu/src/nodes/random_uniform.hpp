// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <random>
#include "kernels/x64/random_uniform.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RandomUniform : public Node {
public:
    union OutputType {
        float    f32;
        float16  f16;
        bfloat16 bf16;
        double   f64;
        int32_t  i32;
        uint32_t u32;
        uint16_t u16;
        int64_t  i64;
    };

    RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    bool needPrepareParams() const override;

    void prepareParams() override;

    void execute(dnnl::stream strm) override;

    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;

    void createPrimitive() override;

    bool created() const override;

    bool canBeInPlace() const override { return false; }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    std::string getPrimitiveDescriptorType() const override;

protected:
    bool needShapeInfer() const override;

private:
    void computeStl(void* out, size_t work_amount);

    std::pair<uint64_t, uint64_t> computePhilox(void* out, size_t work_amount, const std::pair<uint64_t, uint64_t>& prev_state);

    template <typename T, typename DISTR_TYPE>
    void generateData(DISTR_TYPE distribution, void* out, size_t work_amount);

    void initOutShape(VectorDims& dst, const void* src, const element::Type& shape_type, size_t len);

    void initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type);

    void evalRange();

    enum { SHAPE = 0, MIN_VAL, MAX_VAL };
    enum AlgoType { STL, PHILOX };

    bool m_const_inputs[3] = {false, false, false};

    ov::element::Type m_output_prc;
    uint64_t m_global_seed = 0lu;
    uint64_t m_op_seed = 0lu;
    std::pair<uint64_t, uint64_t> m_state {0lu, 0lu};

    VectorDims m_out_shape = {};
    uint64_t m_out_el_num = 1lu;
    OutputType m_min_val;
    OutputType m_max_val;
    OutputType m_range_val;
    AlgoType m_algo = PHILOX;

    std::default_random_engine m_generator;

    struct ThreadParams {
        uint64_t work_amount = 0lu;
        uint64_t dst_shift = 0lu;
        uint64_t n_shift = 0lu;
        uint64_t step = 0lu;
    };

    uint64_t m_threads_num = 0lu;
    std::vector<ThreadParams> m_thread_params;

    ///// PHILOX constants /////

    // Determines how many sequence elements of RNG sequence are skipped between runs.
    // Can be any positive value, 256 is chosen for parity with Tensorflow.
    static constexpr uint64_t SKIP_CONST = 256lu;

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    static constexpr uint64_t PHILOX_GROUP_SIZE = 4lu;

    // Output elements number threshold to execute on one thread.
    static constexpr uint64_t PHILOX_PARALLEL_EXECUTION_THRESHOLD = 1000lu;

    uint64_t m_skip_count = 0lu;
    /////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<kernel::JitKernelBase> m_jit_kernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
