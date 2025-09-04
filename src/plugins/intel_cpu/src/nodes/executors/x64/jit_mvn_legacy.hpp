// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "nodes/executors/mvn_config.hpp"

namespace ov::intel_cpu::legacy {

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
    MVNJitExecutorLegacy(const MVNAttrs& mvnAttrs, const dnnl::primitive_attr& attr);
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

}  // namespace ov::intel_cpu::legacy