// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_deconv.hpp"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

ACLDeconvTensorInfo getACLDeconvTensorInfo(const DeconvAttrs& deconvAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) {
    auto srcDims  = srcDescs[0]->getShape().getDims();
    auto weiDims  = srcDescs[1]->getShape().getDims();
    // swap input and output channels dimensions to be align with ACL
    // weights tensor shape is changed because ACL expects [O, I, H, W] tensor while OV uses [I, O, H, W] tensor
    std::swap(weiDims[0], weiDims[1]);
    auto dstDims  = dstDescs[0]->getShape().getDims();

    VectorDims biasDims;
    TensorInfo biasTensorInfo;

    if (deconvAttrs.withBiasesParam) {
        biasDims = srcDescs[2]->getShape().getStaticDims();
        biasTensorInfo = TensorInfo(shapeCast(biasDims), 1,
                                    precisionToAclDataType(srcDescs[2]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
    }

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
                                          precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo weiTensorInfo = TensorInfo(shapeCast(weiDims), 1,
                                          precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
                                          precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    int pad_l = (deconvAttrs.paddingL.size() > 1) ? static_cast<int>(deconvAttrs.paddingL.at(1)) : static_cast<int>(deconvAttrs.paddingL.at(0));
    int pad_r = (deconvAttrs.paddingR.size() > 1) ? static_cast<int>(deconvAttrs.paddingR.at(1)) : static_cast<int>(deconvAttrs.paddingR.at(0));
    int pad_t = static_cast<int>(deconvAttrs.paddingL.at(0));
    int pad_b = static_cast<int>(deconvAttrs.paddingR.at(0));

    unsigned int stride_x = (deconvAttrs.stride.size() > 1) ? deconvAttrs.stride.at(1) : deconvAttrs.stride.at(0);
    unsigned int stride_y = deconvAttrs.stride.at(0);
    PadStrideInfo deconv_info(stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, DimensionRoundingType::FLOOR);

    return ACLDeconvTensorInfo{srcTensorInfo, weiTensorInfo, biasTensorInfo, dstTensorInfo, deconv_info};
}

AclDeconvExecutor::AclDeconvExecutor(const ExecutorContext::CPtr context) : DeconvExecutor(context) {}

bool AclDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    this->deconvAttrs = deconvAttrs;
    ACLDeconvTensorInfo aclDeconvTensorInfo = getACLDeconvTensorInfo(deconvAttrs, srcDescs, dstDescs);
    TensorInfo srcTensorInfo = aclDeconvTensorInfo.srcTensorInfo;
    TensorInfo weiTensorInfo = aclDeconvTensorInfo.weiTensorInfo;
    TensorInfo biasTensorInfo = aclDeconvTensorInfo.biasTensorInfo;
    TensorInfo dstTensorInfo = aclDeconvTensorInfo.dstTensorInfo;
    PadStrideInfo deconv_info = aclDeconvTensorInfo.deconv_info;

    arm_compute::Status status = arm_compute::NEDeconvolutionLayer::validate(&srcTensorInfo,
                                                                             &weiTensorInfo,
                                                                             deconvAttrs.withBiasesParam ? &biasTensorInfo : nullptr,
                                                                             &dstTensorInfo,
                                                                             deconv_info);
    if (!status) {
        DEBUG_LOG("NEDeconvolutionLayer validation failed: ", status.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);
    if (deconvAttrs.withBiasesParam)
        biasTensor.allocator()->init(biasTensorInfo);

    deconv = std::make_unique<arm_compute::NEDeconvolutionLayer>();
    deconv->configure(&srcTensor, &weiTensor, deconvAttrs.withBiasesParam ? &biasTensor : nullptr, &dstTensor, deconv_info);

    // weights tensor shape is changed because ACL expects [O, I, H, W] tensor while OV uses [I, O, H, W] tensor
     weiBuffer = std::vector<float>(srcDescs[1]->getShape().getStaticDims()[0] *
                                    srcDescs[1]->getShape().getStaticDims()[1] *
                                    srcDescs[1]->getShape().getStaticDims()[2] *
                                    srcDescs[1]->getShape().getStaticDims()[3]);
    return true;
}

static void transpose_to_1023(const MemoryCPtr& srcMemPtr, std::vector<float>& dst_data) {
    const auto src_data = reinterpret_cast<float*>(srcMemPtr->getData());

    const int DIM0 = srcMemPtr->getStaticDims()[0];
    const int DIM1 = srcMemPtr->getStaticDims()[1];
    const int DIM2 = srcMemPtr->getStaticDims()[2];
    const int DIM3 = srcMemPtr->getStaticDims()[3];

    parallel_for3d(DIM0, DIM1, DIM2, [&](const int dim0, const int dim1, const int dim2) {
                for (int dim3 = 0; dim3 < DIM3; ++dim3) {
                    const int src_off = dim0 * DIM1 * DIM2 * DIM3 +
                                        dim1 * DIM2 * DIM3 +
                                        dim2 * DIM3 +
                                        dim3;
                    const int dst_off = dim1 * DIM0 * DIM2 * DIM3 +
                                        dim0 * DIM2 * DIM3 +
                                        dim2 * DIM3 +
                                        dim3;

                    dst_data[dst_off] = src_data[src_off];
                }
    });
}

void AclDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    // TODO: Remove transpose from exec
    transpose_to_1023(src[1], weiBuffer);

    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());
    weiTensor.allocator()->import_memory(weiBuffer.data());
    if (deconvAttrs.withBiasesParam)
        biasTensor.allocator()->import_memory(src[2]->getData());
    deconv->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    weiTensor.allocator()->free();
    if (deconvAttrs.withBiasesParam)
        biasTensor.allocator()->free();
}

bool AclDeconvExecutorBuilder::customIsSupported(const DeconvAttrs &deconvAttrs,
                                                 const std::vector<MemoryDescPtr> &srcDescs,
                                                 const std::vector<MemoryDescPtr> &dstDescs)  {
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

    if (deconvAttrs.withBiasesParam && srcDescs[2]->getPrecision() != srcDescs[0]->getPrecision()) {
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

    if (deconvAttrs.withBiasesParam &&
        !(srcDescs[2]->hasLayoutType(LayoutType::ncsp)) &&
        !(srcDescs[2]->hasLayoutType(LayoutType::nspc))) {
        DEBUG_LOG("AclDeconvExecutor does not support layouts:",
                  " src[0]=", srcDescs[0]->serializeFormat(),
                  " src[1]=", srcDescs[1]->serializeFormat(),
                  " src[2]=", srcDescs[2]->serializeFormat(),
                  " dst=", dstDescs[0]->serializeFormat());
        return false;
    }

    ACLDeconvTensorInfo aclDeconvTensorInfo = getACLDeconvTensorInfo(deconvAttrs, srcDescs, dstDescs);
    TensorInfo srcTensorInfo = aclDeconvTensorInfo.srcTensorInfo;
    TensorInfo weiTensorInfo = aclDeconvTensorInfo.weiTensorInfo;
    TensorInfo biasTensorInfo = aclDeconvTensorInfo.biasTensorInfo;
    TensorInfo dstTensorInfo = aclDeconvTensorInfo.dstTensorInfo;
    PadStrideInfo deconv_info = aclDeconvTensorInfo.deconv_info;

    unsigned int kernel_x = (deconvAttrs.kernel.size() > 1) ? deconvAttrs.kernel.at(1) : deconvAttrs.kernel.at(0);
    unsigned int kernel_y = deconvAttrs.kernel.at(0);

    // After stride=8 up-sampling in ACL Deconvolution layer slower than reference
    if (deconv_info.stride().first >= 8 || deconv_info.stride().second >= 8) return false;

    unsigned int dilation_x = (deconvAttrs.dilation.size() > 1) ? deconvAttrs.dilation.at(1) : deconvAttrs.dilation.at(0);
    unsigned int dilation_y = deconvAttrs.dilation.at(0);
    if (!one_of(dilation_x, static_cast<unsigned int >(0), static_cast<unsigned int >(1)) ||
        !one_of(dilation_y, static_cast<unsigned int >(0), static_cast<unsigned int >(1))) return false;

    size_t in_h = srcDescs[0]->hasLayoutType(LayoutType::ncsp) ? srcDescs[0]->getShape().getDims()[2] : srcDescs[0]->getShape().getDims()[1];
    size_t in_w = srcDescs[0]->hasLayoutType(LayoutType::ncsp) ? srcDescs[0]->getShape().getDims()[3] : srcDescs[0]->getShape().getDims()[2];

    // Validate function has bug (https://github.com/ARM-software/ComputeLibrary/issues/1061) with error exception.
    // We copy deconvolution_output_dimensions function for get correct validation
    // TODO: remove after fix
    if (validate_deconvolution_output_dimensions(in_w, in_h, kernel_x, kernel_y, deconv_info)) {
        DEBUG_LOG("NEDeconvolutionLayer arm_compute::deconvolution_output_dimensions failed");
        return false;
    }

    arm_compute::Status status = arm_compute::NEDeconvolutionLayer::validate(&srcTensorInfo,
                                                                             &weiTensorInfo,
                                                                             deconvAttrs.withBiasesParam ? &biasTensorInfo : nullptr,
                                                                             &dstTensorInfo,
                                                                             deconv_info);
    if (!status) {
        DEBUG_LOG("NEDeconvolutionLayer validation failed: ", status.error_description());
        return false;
    }

    return true;
}

bool AclDeconvExecutorBuilder::validate_deconvolution_output_dimensions(unsigned int in_width, unsigned int in_height,
                                                                   unsigned int kernel_width,
                                                                   unsigned int kernel_height,
                                                                   const PadStrideInfo &pad_stride_info) {
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
}   // namespace intel_cpu
}   // namespace ov
