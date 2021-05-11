// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

struct jit_def_conv_params {
    int ndims;
    int mb;
    int dg;
    int ngroups, ic, oc, oc_padded;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    bool with_bias;
    bool with_sum;
    int nthr;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking;
    int ur_w;
    int ur_w_tail;
    int typesize_in;
    int typesize_off;
    int typesize_bia;
    int typesize_out;
};

struct jit_def_conv_call_args {
    const void *src;
    const void *off;
    const void *filt;
    const void *bias;
    const void *dst;
    const void *buf;
    size_t oh_pos;
};

struct jit_uni_def_conv_kernel {
    void (*ker_)(const jit_def_conv_call_args *);

    void operator()(const jit_def_conv_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_def_conv_kernel(jit_def_conv_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_def_conv_kernel() {}

    virtual void create_ker() = 0;

    jit_def_conv_params jcp_;
};

class MKLDNNDeformableConvolutionNode : public MKLDNNNode {
public:
    MKLDNNDeformableConvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    InferenceEngine::Precision getRuntimePrecision() const override;

private:
    size_t group = 1;
    std::vector<ptrdiff_t> stride = {};
    std::vector<ptrdiff_t> dilation = {};
    std::vector<ptrdiff_t> paddingL = {};

    int deformable_group = 1;

    jit_def_conv_params jcp = {};

    std::shared_ptr<jit_uni_def_conv_kernel> def_conv_kernel = nullptr;

    void executeReference(const float* src, const float* offsets, const float* weights, float* dst,
                          const std::vector<size_t>& src_strides, const std::vector<size_t>& off_strides,
                          const std::vector<size_t>& wei_strides, const std::vector<size_t>& dst_strides);
    void executeOptimized(const float* src, const float* offsets, const float* weights, float* dst,
                          const std::vector<size_t>& src_strides, const std::vector<size_t>& off_strides,
                          const std::vector<size_t>& dst_strides);
};

}  // namespace MKLDNNPlugin

