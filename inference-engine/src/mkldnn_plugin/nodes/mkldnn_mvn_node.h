// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace MKLDNNPlugin {

#define MAX_MVN_INPUTS 3

struct jit_mvn_config_params {
    bool planar_layout;
    bool across_channels;
    bool normalize_variance;
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    int src_data_size;
    int dst_data_size;
    int inputs_number;  // fused eltwise input with same layout and precision
    int C, D, H, W;
};

struct jit_mvn_call_args {
    const void *src[MAX_MVN_INPUTS];
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

class MKLDNNMVNNode;

struct jit_uni_mvn_kernel {
    void (*ker_)(const jit_mvn_call_args *);

    void operator()(const jit_mvn_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, MKLDNNMVNNode& node) : ker_(nullptr), jcp_(jcp), MVNNode(node) {}
    virtual ~jit_uni_mvn_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    MKLDNNMVNNode& MVNNode;
};

class MKLDNNMVNNode : public MKLDNNNode {
public:
    MKLDNNMVNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNMVNNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const MKLDNNNodePtr& node) const;

    static bool checkAxesSuitability(const std::shared_ptr<const ngraph::Node>&);

    void setHasAxesInput(bool hasAxes);

    bool isDepthWiseNode(const MKLDNNNodePtr& node) const;

private:
    void mvn_pln(const std::vector<const uint8_t *>& src_data, uint8_t *dst_data, const InferenceEngine::SizeVector &dims);

    void mvn_blk(const std::vector<const uint8_t *>& src_data, uint8_t *dst_data, const InferenceEngine::SizeVector &dims);

    void mvn_ref(const uint8_t *src_data, uint8_t *dst_data, const InferenceEngine::SizeVector &dims);

    std::tuple<size_t, size_t, size_t, size_t, size_t> get5dShapes(const InferenceEngine::SizeVector& dims);

    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;

    bool across_channels = false;
    bool normalize_variance = true;
    float eps = 1e-9f;
    // Defines way to add epsilon: inside sqrt or outside.
    enum epsType {
        insideSqrt,
        outsideSqrt
    };
    epsType epsMode_;

    bool hasAxesInput = false;

    InferenceEngine::Precision input_prec, output_prec;
    size_t src_data_size, dst_data_size;

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
};

}  // namespace MKLDNNPlugin

