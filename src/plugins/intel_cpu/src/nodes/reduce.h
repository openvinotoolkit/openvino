// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/reduce.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Reduce : public Node {
public:
    Reduce(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    int getFusingAxis() const override;
    bool canFuse(const NodePtr& node) const override;
    bool canBeInPlace() const override {
        return false;
    }

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr);
    inline void reduceKernelProcess(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                    size_t reduce_w = 2, size_t work_batch = 1, const int *tab_idx = NULL);
    inline void reduceKernelPostProcess(uint8_t *out_ptr);
    inline void initDstData(uint8_t *out_ptr, size_t dst_size);
    inline void create_working_memory();
    inline void create_DH_working_memory();
    inline void calcProcessDstDims(std::vector<int64_t> &reduce_axes, const InferenceEngine::SizeVector &dst_dim);
    inline void set_reduce_dim_flags();
    inline void reduce_ref(const float *in_ptr, float *out_ptr);
    void reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func);
    inline void reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);
    void nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights = false);
    void setJITBeyond5D();
    std::vector<int64_t> update_src_dims();
    bool canApplyJIT(const InferenceEngine::Precision &input_prec, const InferenceEngine::Precision &output_prec) const;

    size_t blockLen;
    size_t dst_size;
    size_t prc_size;
    static constexpr size_t REDUCE_DATA = 0;
    static constexpr size_t REDUCE_INDEXES = 1;
    bool jit_beyond_5D = false;
    bool jit_mode = true;
    bool keepDims = true;
    bool is_hybrid_layout = false;
    bool compile_post_kernel = true;
    bool support_split = false;
    bool ReduceDH_opt = false;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t PD, PW;
    size_t srcDataSize, dstDataSize, prcDataSize;
    size_t reduceStride;
    kernel::ReduceLayoutType layout;
    InferenceEngine::Precision outputPrc;
    InferenceEngine::SizeVector srcDims;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;
    std::vector<int64_t> rawAxes;

    kernel::JitReduceConfigParams jcp;

    dnnl::primitive_attr attr;

    std::vector<const void*> postOpsDataPtrs;

    dnnl::memory prc_mem;
    std::vector<uint8_t> vec_reduceDH_prc;

    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReduceCallArgs>> reduceKernel;
    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReducePostCallArgs>> reducePostKernel;

    static const std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>& op, Reduce& node)>> initializers;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
