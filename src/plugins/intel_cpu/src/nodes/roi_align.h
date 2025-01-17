// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnnl_extension_utils.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

enum ROIAlignLayoutType { ncsp, blk, nspc };

enum ROIAlignedMode { ra_asymmetric, ra_half_pixel_for_nn, ra_half_pixel };

struct jit_roi_align_params {
    Algorithm alg;
    ov::element::Type data_prc;
    int data_size;
    ROIAlignLayoutType layout;
    int pooled_h;
    int pooled_w;
};

struct jit_roi_align_call_args {
    // point to srcData for planar
    // point to srcData address list for other layouts
    const void* src;
    const float* weights;
    const float* scale;
    void* buffer;
    void* dst;
    size_t num_samples;
    size_t work_amount;
    size_t src_stride;
};

struct jit_uni_roi_align_kernel {
    void (*ker_)(const jit_roi_align_call_args*);

    void operator()(const jit_roi_align_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_align_kernel(jit_roi_align_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_roi_align_kernel() {}

    virtual void create_ker() = 0;

    jit_roi_align_params jcp_;
};

class ROIAlign : public Node {
public:
    ROIAlign(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    ROIAlignedMode alignedMode;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template <typename T>
    struct ROIAlignExecute;

    void createJitKernel(const ov::element::Type& dataPrec, const ROIAlignLayoutType& selectLayout);
    std::shared_ptr<jit_uni_roi_align_kernel> roi_align_kernel = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
