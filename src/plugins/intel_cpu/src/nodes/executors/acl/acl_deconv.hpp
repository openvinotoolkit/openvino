// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/deconv.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

class AclDeconvExecutor : public DeconvExecutor {
public:
    explicit AclDeconvExecutor(const ExecutorContext::CPtr context);
    bool init(const DeconvAttrs& deconvAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    DeconvAttrs deconvAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor biasTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEDeconvolutionLayer> deconv = nullptr;
};

class AclDeconvExecutorBuilder : public DeconvExecutorBuilder {
public:
    static bool customIsSupported(const DeconvAttrs& deconvAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs) {
        if ((srcDescs[0]->getShape().getDims().size() != 3 && srcDescs[0]->getShape().getDims().size() != 4) ||
             dstDescs[0]->getShape().getDims().size() != srcDescs[0]->getShape().getDims().size() ||
             srcDescs[1]->getShape().getDims().size() != 4) {
            DEBUG_LOG("AclDeconvExecutor does not support dimension:",
                      " src[0]=", srcDescs[0]->getShape().getDims().size(),
                      " src[1]=", srcDescs[1]->getShape().getDims().size(),
                      " dst[0]=", dstDescs[0]->getShape().getDims().size());
            return false;
        }

        // TODO: Ticket CVS-114087 - enable FP16 when check FP16 scoup
        if (!(one_of(srcDescs[0]->getPrecision(), /*InferenceEngine::Precision::FP16, */InferenceEngine::Precision::FP32) &&
              srcDescs[0]->getPrecision() == srcDescs[1]->getPrecision() &&
              srcDescs[1]->getPrecision() == dstDescs[0]->getPrecision())) {
            DEBUG_LOG("AclDeconvExecutor does not support precisions:",
                      " src[0]=", srcDescs[0]->getPrecision(),
                      " src[1]=", srcDescs[1]->getPrecision(),
                      " dst[0]=", dstDescs[0]->getPrecision());
            return false;
        }

        if (deconvAttrs.withBiases && srcDescs[2]->getPrecision() != srcDescs[0]->getPrecision()) {
            DEBUG_LOG("AclDeconvExecutor does not support precisions:",
                      " src[2]=", srcDescs[2]->getPrecision());
            return false;
        }

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            DEBUG_LOG("AclDeconvExecutor does not support layouts:",
                      " src[0]=", srcDescs[0]->serializeFormat(),
                      " src[1]=", srcDescs[1]->serializeFormat(),
                      " dst=", dstDescs[0]->serializeFormat());
            return false;
        }

        if (deconvAttrs.withBiases &&
            !(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[2]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[2]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            DEBUG_LOG("AclDeconvExecutor does not support layouts:",
                      " src[0]=", srcDescs[0]->serializeFormat(),
                      " src[1]=", srcDescs[1]->serializeFormat(),
                      " src[2]=", srcDescs[2]->serializeFormat(),
                      " dst=", dstDescs[0]->serializeFormat());
            return false;
        }

        auto srcDims  = srcDescs[0]->getShape().getDims();
        auto weiDims  = srcDescs[1]->getShape().getDims();
        // swap input and output channels dimensions to be align with ACL
        // weights tensor shape is changed because ACL expects [W, H, I, O] tensor while OV uses [I, O, H, W] tensor
        std::swap(weiDims[0], weiDims[1]);
        auto dstDims  = dstDescs[0]->getShape().getDims();

        VectorDims biasDims;
        arm_compute::TensorInfo biasTensorInfo;
        if (deconvAttrs.withBiases) {
            biasDims = srcDescs[2]->getShape().getStaticDims();
            //bias presicion is I32 but ACL requests bias precision as input ones
            biasTensorInfo = arm_compute::TensorInfo(shapeCast(biasDims), 1,
                                        precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
        }

        arm_compute::TensorInfo srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1,
                                              precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
        arm_compute::TensorInfo weiTensorInfo = arm_compute::TensorInfo(shapeCast(weiDims), 1,
                                              precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
        arm_compute::TensorInfo dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1,
                                              precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

        unsigned int pad_l = (deconvAttrs.paddingL.size() > 1) ? std::abs(deconvAttrs.paddingL.at(1)) : std::abs(deconvAttrs.paddingL.at(0));
        unsigned int pad_r = (deconvAttrs.paddingR.size() > 1) ? std::abs(deconvAttrs.paddingR.at(1)) : std::abs(deconvAttrs.paddingR.at(0));
        unsigned int pad_t = std::abs(deconvAttrs.paddingL.at(0));
        unsigned int pad_b = std::abs(deconvAttrs.paddingR.at(0));
        unsigned int stride_x = (deconvAttrs.stride.size() > 1) ? deconvAttrs.stride.at(1) : deconvAttrs.stride.at(0);
        unsigned int stride_y = deconvAttrs.stride.at(0);
        unsigned int kernel_x = (deconvAttrs.kernel.size() > 1) ? deconvAttrs.kernel.at(1) : deconvAttrs.kernel.at(0);
        unsigned int kernel_y = deconvAttrs.kernel.at(0);

        // After stride=8 up-sampling in ACL Deconvolution layer slower than reference
        if (stride_x >= 8 || stride_y >= 8) return false;

        arm_compute::PadStrideInfo deconv_info(stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR);

        size_t in_h = srcDescs[0]->hasLayoutType(LayoutType::ncsp) ? srcDims[2] : srcDims[1];
        size_t in_w = srcDescs[0]->hasLayoutType(LayoutType::ncsp) ? srcDims[3] : srcDims[2];

        // Validate function has bug (https://github.com/ARM-software/ComputeLibrary/issues/1061) with error exception.
        // We copy deconvolution_output_dimensions function for get correct validation
        // TODO: remove after fix
        if (validate_deconvolution_output_dimensions(in_w, in_h, kernel_x, kernel_y, deconv_info)) {
            DEBUG_LOG("NEDeconvolutionLayer arm_compute::deconvolution_output_dimensions failed");
            return false;
        }

        arm_compute::Status status = arm_compute::NEDeconvolutionLayer::validate(&srcTensorInfo,
                                                                                 &weiTensorInfo,
                                                                                 deconvAttrs.withBiases ? &biasTensorInfo : nullptr,
                                                                                 &dstTensorInfo,
                                                                                 deconv_info);
        if (!status) {
            DEBUG_LOG("NEDeconvolutionLayer validation failed: ", status.error_description());
            return false;
        }

        return true;
    }

    bool isSupported(const DeconvAttrs& deconvAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return customIsSupported(deconvAttrs, srcDescs, dstDescs);
    }

    DeconvExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclDeconvExecutor>(context);
    }

private:
    static bool validate_deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                         unsigned int kernel_width, unsigned int kernel_height,
                                                         const arm_compute::PadStrideInfo &pad_stride_info) {
        const unsigned int pad_left   = pad_stride_info.pad_left();
        const unsigned int pad_top    = pad_stride_info.pad_top();
        const unsigned int pad_right  = pad_stride_info.pad_right();
        const unsigned int pad_bottom = pad_stride_info.pad_bottom();
        const unsigned int stride_x   = pad_stride_info.stride().first;
        const unsigned int stride_y   = pad_stride_info.stride().second;

        if (!((in_width < 1 || in_height < 1) ||
            (((in_width - 1) * stride_x + kernel_width) < (pad_left + pad_right)) ||
            (((in_height - 1) * stride_y + kernel_height) < (pad_top + pad_bottom)))) { return false; }
        return true;
    }
};

}   // namespace intel_cpu
}   // namespace ov
