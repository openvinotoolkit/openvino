// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>
#include "kernels/roi_align_kernel.h"

namespace ov {
namespace intel_cpu {
namespace node {

enum ROIAlignedMode {
    ra_asymmetric,
    ra_half_pixel_for_nn,
    ra_half_pixel
};

class ROIAlign : public Node {
public:
    ROIAlign(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    ROIAlignedMode alignedMode;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template<typename T>
    struct ROIAlignExecute;

    void createJitKernel(const InferenceEngine::Precision& dataPrec, const ROIAlignLayoutType& selectLayout);
    std::shared_ptr<jit_uni_roi_align_kernel> roi_align_kernel = nullptr;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
