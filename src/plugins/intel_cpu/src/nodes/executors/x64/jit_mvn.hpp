// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"
#include "cpu_types.h"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "nodes/executors/mvn_config.hpp"

namespace ov::intel_cpu {

struct jit_mvn_config_params {
    MVNLayoutType layout = mvn_planar;
    bool across_channels = false;
    bool normalize_variance = false;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size = 0;
    int dst_data_size = 0;
};

struct jit_mvn_call_args {
    const void* src;
    void* dst;
    float* sum;
    float* mean;
    float* variance;
    size_t work_amount;
    size_t oc_off;
    // shape need for shape agnostic kernel passed with each infer.
    // OC for block layout and nspc per channel, tails for ncsp and nspc across channel.
    size_t rt_shape_size;
    const void* post_op_data;
};

struct jit_uni_mvn_mean_variance_kernel {
    void (*ker_)(const jit_mvn_call_args*) = nullptr;

    void operator()(const jit_mvn_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_mean_variance_kernel(jit_mvn_config_params jcp) : jcp_(jcp) {}
    virtual ~jit_uni_mvn_mean_variance_kernel() = default;

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
};

struct jit_uni_mvn_kernel {
    void (*ker_)(const jit_mvn_call_args*) = nullptr;

    void operator()(const jit_mvn_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, const dnnl_primitive_attr& attr) : jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_mvn_kernel() = default;

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    const dnnl_primitive_attr& attr_;
    int optimized_scaleshift_num = 0;
};

namespace legacy {

class MVNJitExecutor : public MVNExecutorBase {
public:
    MVNJitExecutor(const MVNAttrs& mvnAttrs, const dnnl::primitive_attr& attr);

    void exec(const uint8_t* src_data,
              uint8_t* dst_data,
              const void* post_ops_data_,
              const VectorDims& shape5d) override;

private:
    void mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
};

}  // namespace legacy

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

class JITMVNExecutor : public Executor {
public:
    JITMVNExecutor(const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr context)
        : jitContext(context),
          jitMVNAttrs(attrs) {
        jitMVNAttrs.postOpsDataPtrs = attrs.postOpsDataPtrs;
    }

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        impl_desc_type impl_type;
        if (mayiuse(cpu::x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(cpu::x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        } else if (mayiuse(cpu::x64::sse41)) {
            impl_type = impl_desc_type::jit_sse42;
        } else {
            impl_type = impl_desc_type::ref;
        }
        return impl_type;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const MVNConfig& config);

private:
    ExecutorContext::CPtr jitContext;
    MVNAttrs jitMVNAttrs;
    VectorDims shape5D;
    std::vector<const void*> postOpsDataPtrs;
    std::shared_ptr<legacy::MVNJitExecutor> oldMVNJitExecutor;

    struct MVNKey {
        MVNAttrs mvnAttrs;
        dnnl::primitive_attr attr;

        size_t hash() const;
        bool operator==(const MVNKey& rhs) const;
    };
};

}  // namespace ov::intel_cpu
