// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <random>

#include "kernels/x64/random_uniform.hpp"

namespace ov::intel_cpu::node {

class RandomUniform : public Node {
public:
    union OutputType {
        double f64;
        float f32;
        float16 f16;
        bfloat16 bf16;
        int64_t i64;
        int32_t i32;
        uint32_t u32;
        uint16_t u16;
    };

    RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    [[nodiscard]] bool needPrepareParams() const override;

    void prepareParams() override;

    void execute(const dnnl::stream& strm) override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;

    void createPrimitive() override;

    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    [[nodiscard]] std::string getPrimitiveDescriptorType() const override;

protected:
    [[nodiscard]] bool needShapeInfer() const override;

private:
    void evalRange();

    void initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type);

    void prepareGeneratorKernel();

    enum PortIndex { SHAPE = 0, MIN_VAL, MAX_VAL };
    enum AlgorithmType { STL = 0, PHILOX, MERSENNE_TWISTER };

    bool m_const_inputs[3] = {false, false, false};

    ov::element::Type m_output_prc;
    uint64_t m_global_seed = 0lu;
    uint64_t m_op_seed = 0lu;
    std::pair<uint64_t, uint64_t> m_state{0lu, 0lu};

    VectorDims m_out_shape = {};
    uint64_t m_output_elements_count = 1lu;
    OutputType m_min_val;
    OutputType m_max_val;
    OutputType m_range_val;
    AlgorithmType m_algo = STL;

    /////////////////////////////////////////////////////////////////////////////////

    ///// PARALLELISM /////

    std::shared_ptr<kernel::JitKernelBase> m_jit_kernel;

    struct PhiloxThreadParams {
        uint64_t work_amount = 0lu;
        uint64_t dst_shift = 0lu;
        uint64_t n_shift = 0lu;
        uint64_t step = 0lu;
    };

    struct MersenneTwisterThreadParams {
        uint64_t src_start_idx = 0lu;
        uint64_t dst_start_idx = 0lu;
        uint64_t state_accesses_count = 0lu;
    };

    int32_t m_threads_num = 0;

    std::vector<PhiloxThreadParams> m_philox_thread_params;
    std::vector<MersenneTwisterThreadParams> m_mersenne_twister_thread_params;

    /////////////////////////////////////////////////////////////////////////////////

    ///// PHILOX /////

    // Output elements number threshold to execute on one thread.
    static constexpr uint64_t PHILOX_PARALLEL_EXECUTION_THRESHOLD = 1000lu;

    // Determines how many sequence elements of RNG sequence are skipped between runs.
    // 256 is chosen for parity with Tensorflow.
    static constexpr uint64_t SKIP_CONST = 256lu;

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    static constexpr uint64_t PHILOX_GROUP_SIZE = 4lu;

    // Used to parallelize state generation
    uint64_t m_skip_count = 0lu;

    void preparePhiloxParams();

    std::pair<uint64_t, uint64_t> computePhilox(void* out,
                                                size_t work_amount,
                                                const std::pair<uint64_t, uint64_t>& prev_state);

    /////////////////////////////////////////////////////////////////////////////////

    ///// MERSENNE TWISTER /////

    // PyTorch reduces the execution time when generating 64-bit numbers when the range is below max value of uint32_t
    // To reduce variable use, value of 'true' denotes the case in which for every uint32_t a single random value is
    // generated for any dtype. Therefore, value of 'false' occurs only when dtype is int64 AND the range is above
    // uint32_t.
    bool m_mersenne_twister_optimization_enabled = true;

    int32_t m_uint_storage_capacity_per_thread = 1;

    void prepareMersenneTwisterParams();

    void computeMersenneTwister(void* out, size_t work_amount);

    /////////////////////////////////////////////////////////////////////////////////

    ///// STL /////

    std::default_random_engine m_generator;

    template <typename T, typename DISTR_TYPE>
    void generateData(DISTR_TYPE distribution, void* out, size_t work_amount);

    void computeStl(void* out, size_t work_amount);
};

}  // namespace ov::intel_cpu::node
