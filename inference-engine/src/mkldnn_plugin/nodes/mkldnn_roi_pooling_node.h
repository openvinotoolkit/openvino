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

enum ROIPoolingOpType {
    Max,
    Bilinear
};

struct jit_roi_pooling_params {
    int mb, c;
    int ih, iw, oh, ow;

    int c_block, nb_c, nb_c_blocking;

    double spatial_scale;
    int pooled_h;
    int pooled_w;

    ROIPoolingOpType alg;
};

struct jit_roi_pooling_call_args {
    const float *src;
    float *dst;

    size_t kh;
    size_t kw;
    size_t bin_area;

    size_t c_blocks;

    float xf;
    float yf;

    size_t xoff;
    size_t yoff;
};

struct jit_uni_roi_pooling_kernel {
    void (*ker_)(const jit_roi_pooling_call_args *);

    void operator()(const jit_roi_pooling_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_pooling_kernel(jit_roi_pooling_params jpp) : ker_(nullptr), jpp_(jpp) {}
    virtual ~jit_uni_roi_pooling_kernel() {}

    virtual void create_ker() = 0;

    jit_roi_pooling_params jpp_;
};

class MKLDNNROIPoolingNode : public MKLDNNNode {
public:
    MKLDNNROIPoolingNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNROIPoolingNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    int pooled_h = 0;
    int pooled_w = 0;
    float spatial_scale = 0;
    ROIPoolingOpType opType = Max;

    jit_roi_pooling_params jpp = {};

    std::shared_ptr<jit_uni_roi_pooling_kernel> roi_pooling_kernel = nullptr;
};

}  // namespace MKLDNNPlugin

