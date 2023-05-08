// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/normalize.hpp"
#include <cpu/ref_eltwise.hpp>
#include <cpu/ref_depthwise_injector.hpp>

namespace ov {
namespace intel_cpu {

typedef std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> ref_eltwise_scalar_fwd_t;
typedef std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> ref_depthwise_scalar_fwd_t;

class RefNormalizeL2Executor : public NormalizeL2Executor {
public:
    explicit RefNormalizeL2Executor(const ExecutorContext::CPtr context);
    bool init(const NormalizeL2Attrs& normalizeL2Attrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) override;
    impl_desc_type getImplType() const override { return normalizeL2Attrs.implDescType; }

private:
    template <typename in_data_t, typename out_data_t>
    static void normalize(const in_data_t* src_data, out_data_t* dst_data, size_t workAmount);

    template <typename in_data_t, typename out_data_t>
    static void normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data, const void **post_ops_data, NormalizeL2Attrs attrs,
                                   dnnl::primitive_attr kernel_attrs,
                                   std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref,
                                   std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref);

    static inline void apply_post_ops_scalar(float &dst_value, int index_c, const void **post_ops_data_,
                                             dnnl::primitive_attr kernel_attrs,
                                             std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref,
                                             std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref,
                                             NormalizeL2Attrs normalizeL2Attrs);

    struct ExecutionContext {
        const uint8_t *src_ptr = nullptr;
        uint8_t *dst_ptr = nullptr;
        const void **post_ops_data = nullptr;
        NormalizeL2Attrs normalizeL2Attrs;
        dnnl::primitive_attr kernel_attrs;
        std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref;
        std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref;
        size_t workAmount = 0lu;
    } execCtx;

    template<typename T>
    struct FunctionExecutorCreation {
        using src_t = typename std::tuple_element<0, T>::type;
        using dst_t = typename std::tuple_element<1, T>::type;

        void operator()(ExecutionContext& ctx) {
            if (ctx.normalizeL2Attrs.cornerCase) {
                normalize(reinterpret_cast<const src_t *>(ctx.src_ptr),
                          reinterpret_cast<dst_t *>(ctx.dst_ptr),
                          ctx.workAmount);
            } else {
                normalize_nchw_ref(reinterpret_cast<const src_t *>(ctx.src_ptr),
                                   reinterpret_cast<dst_t *>(ctx.dst_ptr),
                                   ctx.post_ops_data,
                                   ctx.normalizeL2Attrs, ctx.kernel_attrs,
                                   ctx.eltwise_injectors_ref, ctx.depthwise_injectors_ref);
            }
        }
    };
};

class RefNormalizeL2ExecutorBuilder : public NormalizeL2ExecutorBuilder {
public:
    bool isSupported(const NormalizeL2Attrs& normalizeL2Attrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (!normalizeL2Attrs.cornerCase && normalizeL2Attrs.layout != LayoutType::ncsp)
            return false;
        return true;
    }

    NormalizeL2ExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefNormalizeL2Executor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov