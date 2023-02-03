// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/x64/reduce.hpp"

#include "executors/reduce_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Reduce : public Node {
public:
    Reduce(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

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
    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr);
    inline void reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                    size_t reduce_w = 2, size_t work_batch = 1, const int *tab_idx = NULL);
    inline void reduce_kernel_post_process(uint8_t *out_ptr);
    inline void reduce_kernel_reassign();
    inline void reduce_kernel_restore();
    inline void output_info_reassign(uint8_t *out_ptr);
    inline void output_info_restore(uint8_t **out_ptr);
    inline void init_dst_data(uint8_t *out_ptr, size_t dst_size);
    inline void create_hybrid_working_memory();
    inline void create_opt_working_memory();
    inline void calc_process_dst_dims(std::vector<int64_t> &reduce_axes, const InferenceEngine::SizeVector &dst_dim);
    inline void set_reduce_dim_flags();
    inline void reduce_ref(const float *in_ptr, float *out_ptr);
    void reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func);
    void create_reduce_kernel(std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReduceCallArgs>> &kernel, const kernel::JitReduceConfigParams &jcp);
    inline void reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);
    template<typename T>
    void nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    template<typename T>
    void blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr);
    void setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights = false);
    void setJITBeyond5D();
    std::vector<int64_t> update_src_dims();
    bool canApplyJIT(const InferenceEngine::Precision &input_prec, const InferenceEngine::Precision &output_prec) const;

    size_t blk_size;
    static constexpr size_t REDUCE_DATA = 0;
    static constexpr size_t REDUCE_INDEXES = 1;
    bool jit_beyond_5D = false;
    bool jit_mode = true;
    bool keep_dims = true;
    bool is_hybrid_layout = false;
    bool compile_post_kernel = true;
    bool apply_post_kernel = true;
    bool apply_division = false;
    bool fuse_low_precision = false;
    bool support_split = false;
    bool precision_change = false;
    bool ReduceAll_opt = false;
    bool ReduceDH_opt = false;
    bool ReduceCDW_opt = false;
    bool use_aux_kernel = false;
    bool set_use_aux_kernel = false;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t PD, PH, PW;
    size_t src_data_size, dst_data_size, prc_data_size, intermediate_data_size, tmp_data_size;
    size_t dst_size, prc_size, intermediate_size, tmp_size;
    size_t reduce_stride;
    uint8_t *tmp_ptr;
    kernel::ReduceLayoutType layout;
    InferenceEngine::Precision input_prec, output_prec, intermediate_prec, tmp_prec;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;
    std::vector<int64_t> raw_axes;
    std::vector<uint8_t> intermediate_buf;
    float in_out_divisor_f32 = 1.f;
    double in_out_divisor_f64 = 1.;
    void* in_out_divisor;

    kernel::JitReduceConfigParams jcp;
    kernel::JitReduceConfigParams aux_jcp;

    dnnl::primitive_attr attr;

    std::vector<const void*> postOpsDataPtrs;

    dnnl::memory prc_mem;
    std::vector<uint8_t> vec_reduceDH_prc;
    std::vector<uint8_t> vec_reduceCDW_prc;

    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReduceCallArgs>> reduce_kernel;
    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReduceCallArgs>> reduce_aux_kernel;
    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReduceCallArgs>> reduce_tmp_kernel;
    std::shared_ptr<kernel::JitReduceKernelBase<kernel::JitReducePostCallArgs>> reduce_post_kernel;

    static const std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>& op, Reduce& node)>> initializers;

    std::string errorPrefix;

    ReduceAttrs reduceAttrs;
    bool canUseAclExecutor = false;
    std::shared_ptr<ReduceExecutor> aclExecPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov