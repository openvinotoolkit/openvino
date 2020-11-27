// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

struct jit_resample_config_params {
    bool planar_layout;
    bool nhwc_format;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_resample_call_args {
    const void *src;
    const int *index;
    void *dst;
    size_t src_stride;
    size_t index_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
};

struct jit_uni_resample_nearest_kernel {
    void (*ker_)(const jit_resample_call_args *);

    void operator()(const jit_resample_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_resample_nearest_kernel(jit_resample_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_resample_nearest_kernel() {}

    jit_resample_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

struct jit_uni_resample_linear_kernel {
    void (*ker_)(const jit_resample_call_args *);

    void operator()(const jit_resample_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_resample_linear_kernel(jit_resample_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_resample_linear_kernel() {}

    jit_resample_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};


class MKLDNNResampleNode : public MKLDNNNode {
public:
    MKLDNNResampleNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNResampleNode() override = default;

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
    void NearestNeighbor_PLN(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    template <typename in_data_t, typename out_data_t>
    void NearestNeighbor_BLK(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    template <typename in_data_t, typename out_data_t>
    void LinearInterpolation(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);
    inline void apply_post_ops_scalar(float &dst_value, int index_c);

    int blk_size;

    std::string type;
    bool antialias;
    float factor;

    mkldnn::primitive_attr attr;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    InferenceEngine::Precision input_prec, output_prec;
    size_t src_data_size, dst_data_size;

    std::shared_ptr<jit_uni_resample_nearest_kernel> resample_nearest_kernel;
};

}  // namespace MKLDNNPlugin

