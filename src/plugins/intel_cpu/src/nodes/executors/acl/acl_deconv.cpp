// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_deconv.hpp"
#include "ie_parallel.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclDeconvExecutor::AclDeconvExecutor(const ExecutorContext::CPtr context) : DeconvExecutor(context) {}

bool AclDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    this->deconvAttrs = deconvAttrs;
    auto srcDims  = srcDescs[0]->getShape().getDims();
    auto weiDims  = srcDescs[1]->getShape().getDims();
    // swap input and output channels dimensions to be align with ACL
    // weights tensor shape is changed because ACL expects [W, H, I, O] tensor while OV uses [I, O, H, W] tensor
    std::swap(weiDims[0], weiDims[1]);
    auto dstDims  = dstDescs[0]->getShape().getDims();

    VectorDims biasDims;
    TensorInfo biasTensorInfo;

    if (deconvAttrs.withBiasesParam) {
        biasDims = srcDescs[2]->getShape().getStaticDims();
        //bias presicion is I32 but ACL requests bias precision as input ones
        biasTensorInfo = TensorInfo(shapeCast(biasDims), 1,
        precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
    }

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo weiTensorInfo = TensorInfo(shapeCast(weiDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    unsigned int pad_l = (deconvAttrs.paddingL.size() > 1) ? std::abs(deconvAttrs.paddingL.at(1)) : std::abs(deconvAttrs.paddingL.at(0));
    unsigned int pad_r = (deconvAttrs.paddingR.size() > 1) ? std::abs(deconvAttrs.paddingR.at(1)) : std::abs(deconvAttrs.paddingR.at(0));
    unsigned int pad_t = std::abs(deconvAttrs.paddingL.at(0));
    unsigned int pad_b = std::abs(deconvAttrs.paddingR.at(0));
    unsigned int stride_x = (deconvAttrs.stride.size() > 1) ? deconvAttrs.stride.at(1) : deconvAttrs.stride.at(0);
    unsigned int stride_y = deconvAttrs.stride.at(0);

    arm_compute::PadStrideInfo deconv_info(stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR);
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
    //weights tensor shape is changed because ACL expects [W, H, I, O] tensor while OV uses [I, O, H, W] tensor
    std::vector<float> weiBuffer(src[1]->getStaticDims()[0] *
                                 src[1]->getStaticDims()[1] *
                                 src[1]->getStaticDims()[2] *
                                 src[1]->getStaticDims()[3]);
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

}   // namespace intel_cpu
}   // namespace ov
