// Copyright (C) 2018-2020 Intel Corporation
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

struct jit_mvn_config_params {
    bool planar_layout;
    bool across_channels;
    bool normalize_variance;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
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
    MKLDNNMVNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNMVNNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    template <typename in_data_t, typename out_data_t>
    void mvn_pln(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    template <typename in_data_t, typename out_data_t>
    void mvn_blk(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);

    std::tuple<size_t, size_t, size_t, size_t, size_t> get5dShapes(const InferenceEngine::SizeVector& dims);

    bool across_channels = false;
    bool normalize_variance = true;
    float eps = 1e-9f;

    InferenceEngine::Precision input_prec, output_prec;
    size_t src_data_size, dst_data_size;

    mkldnn::primitive_attr attr;

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;
};

}  // namespace MKLDNNPlugin

