// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"

namespace ov::intel_cpu {

namespace legacy {

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

class MVNJitExecutorLegacy {
public:
    MVNJitExecutorLegacy(const MVNAttrs& mvnAttrs,
                         const dnnl::primitive_attr& attr,
                         ov::element::Type src_prc,
                         ov::element::Type dst_prc);
    void exec(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);

private:
    void mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);
    void mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void* post_ops_data_, const VectorDims& shape5d);

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;

    MVNAttrs mvnAttrs;
    size_t src_data_size = 0;
    size_t dst_data_size = 0;
};

}  // namespace legacy

class MVNJitExecutor : public Executor {
public:
    MVNJitExecutor(MVNAttrs mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;

    void execute() override {
        executeImpl(memoryArgs);
    }

    void execute(const MemoryArgs& memory) override {
        executeImpl(memory);
    }

    void executeImpl(const MemoryArgs& memory);

    [[nodiscard]] impl_desc_type implType() const override {
        // Return a specific ISA implementation type based on runtime capabilities
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            return impl_desc_type::jit_avx512;
        }
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            return impl_desc_type::jit_avx2;
        }
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
            return impl_desc_type::jit_sse42;
        }
        return impl_desc_type::ref;
    }

    static bool supports(const MVNConfig& config);

    bool canReuseShapeAgnosticKernel(const VectorDims& newShape5D);

private:
    void setPostOps(dnnl::primitive_attr& attr, bool initWeights = false);

    MVNAttrs attrs;
    std::vector<const void*> postOpsDataPtrs;
    std::vector<uint8_t> postOpsDataBuffer;  // Continuous buffer for legacy implementation
    std::vector<void*> postOpsPtrArray;      // Array of pointers for legacy MVN
    std::vector<MemoryPtr> postOpsMemory;    // Keep memory alive
    MemoryArgs memoryArgs;
    const ExecutorContext::CPtr context;
    VectorDims shape5D;
    std::shared_ptr<legacy::MVNJitExecutorLegacy> legacyJitExecutor;
};

}  // namespace ov::intel_cpu
