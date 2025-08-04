// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

enum ROIAlignLayoutType : uint8_t { ncsp, blk, nspc };

enum ROIAlignedMode : uint8_t { ra_asymmetric, ra_half_pixel_for_nn, ra_half_pixel };

struct jit_roi_align_params {
    Algorithm alg = Algorithm::ROIAlignMax;
    ov::element::Type data_prc;
    int data_size = 0;
    ROIAlignLayoutType layout = ncsp;
    int pooled_h = 0;
    int pooled_w = 0;
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
    void (*ker_)(const jit_roi_align_call_args*) = nullptr;

    void operator()(const jit_roi_align_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_roi_align_kernel(jit_roi_align_params jcp) : jcp_(jcp) {}
    virtual ~jit_uni_roi_align_kernel() = default;

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
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0F;
    ROIAlignedMode alignedMode;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template <typename T>
    struct ROIAlignExecute;

    void createJitKernel(const ov::element::Type& dataPrec, const ROIAlignLayoutType& selectLayout);
    std::shared_ptr<jit_uni_roi_align_kernel> roi_align_kernel = nullptr;
};

}  // namespace ov::intel_cpu::node
