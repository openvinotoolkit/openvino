// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "executors/mvn_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

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
    void (*ker_)(const jit_mvn_call_args*);

    void operator()(const jit_mvn_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_mean_variance_kernel(jit_mvn_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_mvn_mean_variance_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
};

struct jit_uni_mvn_kernel {
    void (*ker_)(const jit_mvn_call_args*);

    void operator()(const jit_mvn_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, const dnnl_primitive_attr& attr)
        : ker_(nullptr),
          jcp_(jcp),
          attr_(attr) {}
    virtual ~jit_uni_mvn_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    const dnnl_primitive_attr& attr_;
    int optimized_scaleshift_num = 0;
};

class MVN : public Node {
public:
    MVN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    inline bool getAcrossChannels() const {
        return mvnAttrs.initAcrossChannels_;
    }

    inline bool getNormalizeVariance() const {
        return mvnAttrs.normalizeVariance_;
    }

    bool canFuse(const NodePtr& node) const override;
    void prepareParams() override;

private:
    void setPostOps(dnnl::primitive_attr& attr, bool initWeights = false);

    void transformTo5DCase(const VectorDims& shape);

    std::vector<const void*> postOpsDataPtrs;

    MVNAttrs mvnAttrs;
    VectorDims shape5D = {0, 0, 0, 0, 0};
    bool onlyUnaryPostOps = true;

    class MVNExecutorBase {
    public:
        MVNExecutorBase(const MVNAttrs& mvnAttrs);
        virtual void exec(const uint8_t* in_ptr_,
                          uint8_t* out_ptr_,
                          const void* post_ops_data_,
                          const VectorDims& shape5d) = 0;
        virtual ~MVNExecutorBase() = default;

    protected:
        MVNAttrs mvnAttrs;
        size_t src_data_size = 0;
        size_t dst_data_size = 0;
    };

    std::shared_ptr<MVNExecutorBase> execPtr = nullptr;
    bool canUseAclExecutor = false;
    std::shared_ptr<MVNExecutor> aclExecPtr = nullptr;

    class MVNJitExecutor : public MVNExecutorBase {
    public:
        MVNJitExecutor(const MVNAttrs& mvnAttrs, const dnnl::primitive_attr& attr);

        void exec(const uint8_t* in_ptr_,
                  uint8_t* out_ptr_,
                  const void* post_ops_data_,
                  const VectorDims& shape5d) override;

    private:
        void mvn_pln(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, const VectorDims& shape5d);
        void mvn_blk(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, const VectorDims& shape5d);
        void mvn_nspc(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_, const VectorDims& shape5d);

        std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
        std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
        std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
    };

    class MVNRefExecutor : public MVNExecutorBase {
    public:
        MVNRefExecutor(const MVNAttrs& mvnAttrs);

        void exec(const uint8_t* in_ptr_,
                  uint8_t* out_ptr_,
                  const void* post_ops_data_,
                  const VectorDims& shape5d) override;

    private:
        void mvn_ref(const uint8_t* in_ptr_, uint8_t* out_ptr_, const VectorDims& shape5d);
    };
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
