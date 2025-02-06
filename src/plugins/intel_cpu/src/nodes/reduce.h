// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/reduce_list.hpp"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

enum ReduceLayoutType { reduce_ncsp, reduce_nspc, reduce_blocked };

struct jit_reduce_config_params {
    ReduceLayoutType layout;
    Algorithm reduce_mode;
    bool fuse_low_precision;
    bool fuse_broadcast;  // if post ops fusion needs broadcast
    bool round_to_zero;
    dnnl::memory::data_type src_dt;
    dnnl::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_reduce_call_args {
    const void* src;
    const int* idx;
    void* dst;
    size_t work_amount;
    size_t work_batch;
    size_t reduce_w =
        2;  // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_stride;  // only used in planar layout while reducing dimensions except for width
    size_t can_divide;     // if apply division in reduce_kernel [1: Yes] [0: No]
    const float* divisor;  // mean = sum / divisor
};

struct jit_reduce_post_call_args {
    const void* src;
    void* dst;
    size_t work_amount;
    size_t reduce_c =
        2;  // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    size_t oc_off;         // offset in byte along channel on output tensor
    size_t channel_size;   // only for post ops fusion of nspc layout
    const float* divisor;  // mean = sum / divisor
    const void** post_op_data;
};

struct jit_uni_reduce_kernel {
    void (*ker_)(const jit_reduce_call_args*);

    void operator()(const jit_reduce_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
};

struct jit_uni_reduce_post_kernel {
    void (*ker_)(const jit_reduce_post_call_args*);

    void operator()(const jit_reduce_post_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_post_kernel(jit_reduce_config_params jcp, const dnnl_primitive_attr& attr)
        : ker_(nullptr),
          jcp_(jcp),
          attr_(attr) {}
    virtual ~jit_uni_reduce_post_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
    const dnnl_primitive_attr& attr_;
};

class Reduce : public Node {
public:
    Reduce(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    int getFusingAxis() const override;
    bool canFuse(const NodePtr& node) const override;
    bool canBeInPlace() const override {
        return false;
    }

    bool neverExecute() const override;
    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    void reduce_type(const uint8_t* in_ptr, uint8_t* out_ptr);
    void reduce_PLN(const uint8_t* in_ptr, uint8_t* out_ptr);
    void reduce_BLK(const uint8_t* in_ptr, uint8_t* out_ptr);
    void reduce_BLK_concern_padding(const uint8_t* in_ptr, uint8_t* out_ptr);
    inline void reduce_kernel_process(const uint8_t* in_p,
                                      uint8_t* out_p,
                                      size_t work_amount,
                                      size_t reduce_w = 2,
                                      size_t work_batch = 1,
                                      const int* tab_idx = NULL);
    inline void reduce_kernel_post_process(uint8_t* out_ptr);
    inline void reduce_kernel_reassign();
    inline void reduce_kernel_restore();
    inline void output_info_reassign(uint8_t** out_ptr);
    inline void output_info_restore(uint8_t** out_ptr);
    inline void init_dst_data(uint8_t* out_ptr, size_t dst_size);
    inline void create_hybrid_working_memory();
    inline void create_opt_working_memory();
    inline void calc_process_dst_dims(std::vector<int>& reduce_axes, const VectorDims& dst_dim);
    inline void set_reduce_dim_flags();
    inline void reduce_ref(const float* in_ptr, float* out_ptr);
    void reduce_ref_process(const float* in_ptr,
                            float* out_ptr,
                            float init_value,
                            std::function<float(float, float)> func);
    void create_reduce_kernel(std::shared_ptr<jit_uni_reduce_kernel>& kernel, const jit_reduce_config_params& jcp);
    inline void reduce_ref_map(float* out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);
    void nspc2ncsp(uint8_t* proc_ptr, uint8_t* out_ptr);
    void blocked2ncsp(uint8_t* proc_ptr, uint8_t* out_ptr);
    void setPostOps(dnnl::primitive_attr& attr, const VectorDims& postOpDims, bool initWeights = false);
    void setJITBeyond5D();
    std::vector<int> update_src_dims();
    bool canApplyJIT(const ov::element::Type& input_prec, const ov::element::Type& output_prec) const;

    size_t blk_size;
    static const size_t REDUCE_DATA = 0;
    static const size_t REDUCE_INDEXES = 1;
    bool jit_beyond_5D = false;
    bool jit_mode = true;
    bool keep_dims = true;
    bool round_to_zero = false;
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
    bool empty_input = false;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t PD, PH, PW;
    size_t src_data_size, dst_data_size, prc_data_size, intermediate_data_size, tmp_data_size;
    size_t dst_size, prc_size, intermediate_size, tmp_size;
    size_t reduce_stride;
    uint8_t* tmp_ptr;
    ReduceLayoutType layout;
    ov::element::Type input_prec, output_prec, intermediate_prec, tmp_prec;
    VectorDims src_dims;
    VectorDims process_dst_dims;
    VectorDims axes_for_reduction;
    std::vector<int> raw_axes;
    std::vector<uint8_t> intermediate_buf;

    jit_reduce_config_params jcp;
    jit_reduce_config_params aux_jcp;

    dnnl::primitive_attr attr;

    std::vector<const void*> postOpsDataPtrs;

    dnnl::memory prc_mem;
    std::vector<uint8_t> vec_reduceDH_prc;
    std::vector<uint8_t> vec_reduceCDW_prc;

    std::shared_ptr<jit_uni_reduce_kernel> reduce_kernel;
    std::shared_ptr<jit_uni_reduce_kernel> reduce_aux_kernel;
    std::shared_ptr<jit_uni_reduce_kernel> reduce_tmp_kernel;
    std::shared_ptr<jit_uni_reduce_post_kernel> reduce_post_kernel;

    static const std::map<const ov::DiscreteTypeInfo,
                          std::function<void(const std::shared_ptr<ov::Node>& op, Reduce& node)>>&
    getInitializers();

#if defined(OV_CPU_WITH_ACL)
    ReduceAttrs reduceAttrs;
    bool canUseAclExecutor = false;
    std::shared_ptr<ReduceExecutor> aclExecPtr = nullptr;
#endif
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
