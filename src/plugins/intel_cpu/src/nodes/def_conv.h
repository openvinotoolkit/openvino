// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

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
    int nthr;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking;
    int ur_w;
    int ur_w_tail;
    int typesize_in;
    int typesize_off;
    int typesize_sampled_wei;
    int typesize_sampled_offsets;
    int typesize_bia;
    int typesize_out;
    bool with_bias;
    bool with_sum;
    bool with_modulation;
    bool with_bi_pad;
};

struct jit_def_conv_call_args {
    const void* src;
    const void* sampledWei;
    const void* sampledCoords;
    const void* filt;
    const void* bias;
    const void* dst;
    const void* buf;
    size_t oh_pos;
};

struct jit_uni_def_conv_kernel {
    void (*ker_)(const jit_def_conv_call_args*);

    void operator()(const jit_def_conv_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_def_conv_kernel(const jit_def_conv_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_def_conv_kernel() {}

    virtual void create_ker() = 0;

    jit_def_conv_params jcp_;
};

class DeformableConvolution : public Node {
public:
    DeformableConvolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    bool enforceRef = false;
    constexpr static int sampledPointsPerPixel = 4;  // count of sampling points ({top|bottom}, {left|right})

    ov::element::Type getRuntimePrecision() const override;

    struct DefConvAttr {
        size_t group = 1;
        int deformable_group = 1;
        bool with_bilinear_pad = false;
        std::vector<ptrdiff_t> stride = {};
        std::vector<ptrdiff_t> dilation = {};
        std::vector<ptrdiff_t> padL;
    } defConvAttr;

private:
    std::vector<int> sampledCoordsVector;
    std::vector<float> interpWeightsVector;

    void prepareParams() override;
    void updatePadding();

    void executeDynamicImpl(const dnnl::stream& strm) override;
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t OFF_ID = 1;
    static constexpr size_t WEI_ID = 2;
    static constexpr size_t MOD_ID = 3;
    class DefConvExecutor {
    public:
        DefConvExecutor(const DefConvAttr& defConvAttr,
                        const std::vector<std::shared_ptr<BlockedMemoryDesc>>& descVector);

        virtual void exec(const float* src,
                          const float* offsets,
                          const float* weights,
                          const float* modulation,
                          float* dst,
                          int* pSampledCoordsVector,
                          float* pInterpWeightsVector) = 0;
        virtual ~DefConvExecutor() = default;

    protected:
        void prepareSamplingWeights(const float* offsets, const float* modulation = nullptr, bool enforceRef = false);
        jit_def_conv_params jcp = {};
        VectorDims srcStrides;
        VectorDims offStrides;
        VectorDims weiStrides;
        VectorDims modStrides;
        VectorDims dstStrides;
        int* pSampledCoordsVector;
        float* pInterpWeightsVector;
    };

    class DefConvRefExecutor : public DefConvExecutor {
    public:
        DefConvRefExecutor(const DefConvAttr& defConvAttr,
                           const std::vector<std::shared_ptr<BlockedMemoryDesc>>& descVector)
            : DefConvExecutor(defConvAttr, descVector) {}

        void exec(const float* src,
                  const float* offsets,
                  const float* weights,
                  const float* modulation,
                  float* dst,
                  int* pSampledCoordsVector,
                  float* pInterpWeightsVector) override;
    };

    class DefConvJitExecutor : public DefConvExecutor {
        std::shared_ptr<jit_uni_def_conv_kernel> def_conv_kernel = nullptr;

    public:
        DefConvJitExecutor(const DefConvAttr& defConvAttr,
                           const std::vector<std::shared_ptr<BlockedMemoryDesc>>& descVector);

        void exec(const float* src,
                  const float* offsets,
                  const float* weights,
                  const float* modulation,
                  float* dst,
                  int* pSampledCoordsVector,
                  float* pInterpWeightsVector) override;
    };

    std::shared_ptr<DefConvExecutor> execPtr = nullptr;
    bool autoPadding = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
