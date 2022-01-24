// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace MKLDNNPlugin {

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
    const float *eps;
    float *size;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
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

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_mvn_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

class MKLDNNMVNNode : public MKLDNNNode {
public:
    MKLDNNMVNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    inline bool getAcrossChannels() const {
        return initAcrossChannels_;
    }

    inline bool getNormalizeVariance() const {
        return normalizeVariance_;
    }

    bool canFuse(const MKLDNNNodePtr& node) const override;

    void prepareParams() override;

private:
    void mvn_pln(const uint8_t *src_data, uint8_t *dst_data);

    void mvn_blk(const uint8_t *src_data, uint8_t *dst_data);

    void mvn_ref(const uint8_t *src_data, uint8_t *dst_data);

    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);

    void transformTo5DCase(const InferenceEngine::SizeVector& shape);

    std::tuple<size_t, size_t, size_t, size_t, size_t> shape5D;

    bool initAcrossChannels_ = false;
    bool execAcrossChannels_ = false;
    bool normalizeVariance_ = true;
    float epsValue_ = 1e-9f;
    // Defines way to add epsilon: inside sqrt or outside.
    enum MVNEpsMode {
        INSIDE_SQRT,
        OUTSIDE_SQRT
    };
    MVNEpsMode epsMode_;

    InferenceEngine::Precision input_prec, output_prec;
    size_t src_data_size = 0;
    size_t dst_data_size = 0;

    mkldnn::primitive_attr attr;

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
};

}  // namespace MKLDNNPlugin

