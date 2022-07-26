// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_mvn_config_params {
    bool planar_layout;
    bool across_channels;
    bool normalize_variance;
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    int src_data_size;
    int dst_data_size;
    int C, D, H, W;
};

struct jit_mvn_call_args {
    const void *src;
    void *dst;
    float *sum;
    float *mean;
    float *variance;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
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
};

class MVN : public Node {
public:
    MVN(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
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

    // Defines way to add epsilon: inside sqrt or outside.
    enum MVNEpsMode {
        INSIDE_SQRT,
        OUTSIDE_SQRT
    };

    enum MVNLayoutType {
        planar,
        block,
        by_channel
    };
    struct MVNAttrs {
        MVNLayoutType layout;
        std::tuple<size_t, size_t, size_t, size_t, size_t> shape5D;
        bool initAcrossChannels_;
        bool execAcrossChannels_;
        bool normalizeVariance_;
        float epsValue_;
        MVNEpsMode epsMode_;
        InferenceEngine::Precision src_prc;
        InferenceEngine::Precision dst_prc;
    };

private:
    void setPostOps(dnnl::primitive_attr &attr, bool initWeights = false);

    void transformTo5DCase(const InferenceEngine::SizeVector& shape);

    std::vector<const void*> postOpsDataPtrs;

    MVNAttrs mvnAttrs;

    class MVNExecutor {
    public:
        MVNExecutor(const MVNAttrs& mvnAttrs);
        virtual void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) = 0;
        virtual ~MVNExecutor() = default;

    protected:
        MVNAttrs mvnAttrs;
        size_t src_data_size = 0;
        size_t dst_data_size = 0;
    };

    std::shared_ptr<MVNExecutor> execPtr = nullptr;

    class MVNJitExecutor : public MVNExecutor {
        public:
            MVNJitExecutor(const MVNAttrs& mvnAttrs,
                           const dnnl::primitive_attr &attr);

            void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

        private:
            void mvn_pln(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_);
            void mvn_blk(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_);

            std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
            std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
            std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
    };

    class MVNRefExecutor : public MVNExecutor {
        public:
            MVNRefExecutor(const MVNAttrs& mvnAttrs);

            void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_) override;

        private:
            void mvn_ref(const uint8_t *in_ptr_, uint8_t *out_ptr_);
    };
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
