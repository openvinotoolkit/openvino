// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/mvn.hpp"
#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise.hpp"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

struct jit_mvn_config_params {
    MVNLayoutType layout;
    bool across_channels;
    bool normalize_variance;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size;
    int dst_data_size;
};

struct jit_mvn_call_args {
    const void *src;
    void *dst;
    float *sum;
    float *mean;
    float *variance;
    size_t work_amount;
    size_t oc_off;
    // shape need for shape agnostic kernel passed with each infer.
    // OC for block layout and nspc per channel, tails for ncsp and nspc across channel.
    size_t rt_shape_size;
    const void* post_op_data;
};

struct jit_uni_mvn_mean_variance_kernel {
    void (*ker_)(const jit_mvn_call_args *);

    void operator()(const jit_mvn_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_mean_variance_kernel(jit_mvn_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_mvn_mean_variance_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
};

struct jit_uni_mvn_kernel {
    void (*ker_)(const jit_mvn_call_args *);

    void operator()(const jit_mvn_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, const dnnl_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_mvn_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    const dnnl_primitive_attr &attr_;
    int optimized_scaleshift_num = 0;
};

class MVNJitExecutor : public MVNExecutorBase {
public:
    MVNJitExecutor(const MVNAttrs& mvnAttrs,
                   const dnnl::primitive_attr &attr);

    void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, const VectorDims& shape5d) override;

private:
    void mvn_pln(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, const VectorDims& shape5d);
    void mvn_blk(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, const VectorDims& shape5d);
    void mvn_nspc(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, const VectorDims& shape5d);

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
};

}   // namespace intel_cpu
}   // namespace ov