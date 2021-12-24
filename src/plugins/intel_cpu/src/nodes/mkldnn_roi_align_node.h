// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <mkldnn_extension_utils.h>

namespace MKLDNNPlugin {

struct jit_roi_align_params {
    Algorithm alg;
    InferenceEngine::Precision data_prc;
    int data_size;
};

struct jit_roi_align_call_args {
    explicit jit_roi_align_call_args(const void *src_, const int *idx_y_, const int *idx_x_, const int *stride_y_, const int *stride_x_,
        const float *weights_, const float *scale_, void *dst_, size_t work_amount_) : src(src_), idx_y(idx_y_), idx_x(idx_x_),
        stride_y(stride_y_), stride_x(stride_x_), weights(weights_), scale(scale_), dst(dst_), work_amount(work_amount_) {}
    const void *src;
    const int *idx_y;
    const int *idx_x;
    const int *stride_y;
    const int *stride_x;
    const float *weights;
    const float *scale;
    void *dst;
    size_t work_amount;
};

struct jit_uni_roi_align_kernel {
    void (*ker_)(const jit_roi_align_call_args *);

    void operator()(const jit_roi_align_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_align_kernel(jit_roi_align_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_roi_align_kernel() {}

    virtual void create_ker() = 0;

    jit_roi_align_params jcp_;
};

class MKLDNNROIAlignNode : public MKLDNNNode {
public:
    MKLDNNROIAlignNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template<typename T>
    struct ROIAlignExecute;

    void createJitKernel(const InferenceEngine::Precision& dataPrec);
    std::shared_ptr<jit_uni_roi_align_kernel> roi_align_kernel = nullptr;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
