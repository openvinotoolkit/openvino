// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/normalize.hpp"
#include <cpu/ref_eltwise.hpp>
#include <cpu/ref_depthwise_injector.hpp>
#include "emitters/x64/jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {

struct jit_normalize_config_params {
    bool is_nchw;
    bool is_nhwc;
    bool is_blk;
    bool across_spatial;
    dnnl::memory::data_type src_dt;
    dnnl::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
    size_t n, c, h, w;
};

struct jit_normalize_call_args {
    const void *src;
    void *dst;
    const float *modulo;
    const float *fused_factor;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
    //ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;
};

struct jit_uni_normalize_modulo_kernel {
    void (*ker_)(const jit_normalize_call_args *);

    void operator()(const jit_normalize_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_normalize_modulo_kernel(jit_normalize_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_normalize_modulo_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
};

struct jit_uni_normalize_kernel {
    void (*ker_)(const jit_normalize_call_args *);

    void operator()(const jit_normalize_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_normalize_kernel(jit_normalize_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_normalize_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
    const dnnl_primitive_attr &attr_;
};

typedef std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> ref_eltwise_scalar_fwd_t;
typedef std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> ref_depthwise_scalar_fwd_t;

class JitNormalizeL2Executor : public NormalizeL2Executor {
public:
    explicit JitNormalizeL2Executor(const ExecutorContext::CPtr context);
    bool init(const NormalizeL2Attrs& normalizeL2Attrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) override;
    impl_desc_type getImplType() const override { return normalizeL2Attrs.implDescType; }

private:
    template <typename in_data_t, typename out_data_t>
    static void normalize_nchw(const in_data_t* src_data, out_data_t* dst_data, const void **post_ops_data, NormalizeL2Attrs attrs,
                               jit_normalize_config_params jcp, size_t blk_size,
                               std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel,
                               std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel);

    template <typename in_data_t, typename out_data_t>
    static void normalize_nhwc(const in_data_t* src_data, out_data_t* dst_data, const void **post_ops_data, NormalizeL2Attrs attrs,
                               jit_normalize_config_params jcp, size_t blk_size,
                               std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel,
                               std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel);

    template <typename in_data_t, typename out_data_t>
    static void normalize_blk(const in_data_t* src_data, out_data_t* dst_data, const void **post_ops_data, NormalizeL2Attrs attrs,
                              jit_normalize_config_params jcp, size_t blk_size,
                              std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel,
                              std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel);

    struct ExecutionContext {
        const uint8_t *src_ptr = nullptr;
        uint8_t *dst_ptr = nullptr;
        const void **post_ops_data = nullptr;
        NormalizeL2Attrs normalizeL2Attrs;
        size_t blk_size = 1lu;
        jit_normalize_config_params jcp = {};
        NormalizeL2Attrs attrs;

        std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel;
        std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel;
    } execCtx;

    template<typename T>
    struct FunctionExecutorCreation {
        using src_t = typename std::tuple_element<0, T>::type;
        using dst_t = typename std::tuple_element<1, T>::type;

        void operator()(ExecutionContext& ctx) {
            if (ctx.jcp.is_nchw) {
                normalize_nchw(reinterpret_cast<const src_t*>(ctx.src_ptr), reinterpret_cast<dst_t*>(ctx.dst_ptr), ctx.post_ops_data,
                               ctx.normalizeL2Attrs, ctx.jcp, ctx.blk_size, ctx.normalize_modulo_kernel, ctx.normalize_kernel);
            } else if (ctx.jcp.is_nhwc) {
                normalize_nhwc(reinterpret_cast<const src_t*>(ctx.src_ptr), reinterpret_cast<dst_t*>(ctx.dst_ptr), ctx.post_ops_data,
                               ctx.normalizeL2Attrs, ctx.jcp, ctx.blk_size, ctx.normalize_modulo_kernel, ctx.normalize_kernel);
            } else if (ctx.jcp.is_blk) {
                normalize_blk(reinterpret_cast<const src_t*>(ctx.src_ptr), reinterpret_cast<dst_t*>(ctx.dst_ptr), ctx.post_ops_data,
                              ctx.normalizeL2Attrs, ctx.jcp, ctx.blk_size, ctx.normalize_modulo_kernel, ctx.normalize_kernel);
            }
        }
    };
};

class JitNormalizeL2ExecutorBuilder : public NormalizeL2ExecutorBuilder {
public:
    bool isSupported(const NormalizeL2Attrs& normalizeL2Attrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41))
            return false;
        return true;
    }

    NormalizeL2ExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<JitNormalizeL2Executor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov