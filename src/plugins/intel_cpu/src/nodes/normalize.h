// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cassert>
#include <cpu/ref_depthwise_injector.hpp>
#include <cpu/ref_eltwise.hpp>

#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "utils/cpu_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
#if defined(OPENVINO_ARCH_X86_64)
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
    const void* src;
    void* dst;
    const float* modulo;
    const float* fused_factor;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
    // ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;
};

struct jit_uni_normalize_modulo_kernel {
    void (*ker_)(const jit_normalize_call_args*);

    void operator()(const jit_normalize_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_normalize_modulo_kernel(jit_normalize_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_normalize_modulo_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
};

struct jit_uni_normalize_kernel {
    void (*ker_)(const jit_normalize_call_args*);

    void operator()(const jit_normalize_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_normalize_kernel(jit_normalize_config_params jcp, const dnnl_primitive_attr& attr)
        : ker_(nullptr),
          jcp_(jcp),
          attr_(attr) {}
    virtual ~jit_uni_normalize_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
    const dnnl_primitive_attr& attr_;
};
#endif
class NormalizeL2 : public Node {
public:
    NormalizeL2(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    bool neverExecute() const override;
    bool isExecutable() const override;

    enum class NormEpsMode { ADD, MAX };

    struct NormalizeL2Attrs {
        LayoutType layout = LayoutType::ncsp;
        NormEpsMode epsMode = NormEpsMode::ADD;
        bool across_spatial = true;
        bool cornerCase = false;
        float eps = 1e-10f;

        ov::element::Type input_prec = ov::element::dynamic;
        ov::element::Type output_prec = ov::element::dynamic;
        size_t src_data_size = 0lu;
        size_t dst_data_size = 0lu;
    };

private:
    NormalizeL2Attrs attrs;

    class NormalizeL2Executor {
    public:
        NormalizeL2Executor() = default;
        virtual void exec(const uint8_t* src_ptr, uint8_t* dst_ptr, const void** post_ops_data) = 0;
        virtual ~NormalizeL2Executor() = default;

        static std::shared_ptr<NormalizeL2Executor> getNormalizeL2Executor(const NormalizeL2Attrs& attrs,
                                                                           const dnnl::primitive_attr& kernel_attr,
                                                                           const VectorDims& dims);

    protected:
        inline float epsApply(const float& modulo, const NormEpsMode mode, const float eps) const {
            return mode == NormEpsMode::ADD ? modulo + eps : std::max(modulo, eps);
        }

    private:
        template <typename in_data_t, typename out_data_t>
        static std::shared_ptr<NormalizeL2Executor> makeExecutor(const NormalizeL2Attrs& attrs,
                                                                 const dnnl::primitive_attr& kernel_attrs,
                                                                 const VectorDims& dims);

        struct NormalizeContext {
            std::shared_ptr<NormalizeL2Executor> executor;
            NormalizeL2Attrs attrs;
            dnnl::primitive_attr kernel_attrs;
            VectorDims dims;
        };

        template <typename T>
        struct NormalizeExecutorCreation {
            using src_t = typename std::tuple_element<0, T>::type;
            using dst_t = typename std::tuple_element<1, T>::type;

            void operator()(NormalizeContext& ctx) {
                ctx.executor = NormalizeL2Executor::makeExecutor<src_t, dst_t>(ctx.attrs, ctx.kernel_attrs, ctx.dims);
            }
        };
    };

    template <typename in_data_t, typename out_data_t>
    class NormalizeL2CornerCaseExecutor;
    template <typename in_data_t, typename out_data_t>
    class NormalizeL2JitExecutor;
    template <typename in_data_t, typename out_data_t>
    class NormalizeL2ReferenceExecutor;

    dnnl::primitive_attr kernel_attrs;

    std::vector<const void*> postOpsDataPtrs;

    void setPostOps(dnnl::primitive_attr& kernel_attrs, const VectorDims& dims, bool initWeights = false);

    static constexpr size_t DATA = 0;
    static constexpr size_t AXES = 1;

    using executorPtr = std::shared_ptr<NormalizeL2Executor>;
    executorPtr execPtr = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
