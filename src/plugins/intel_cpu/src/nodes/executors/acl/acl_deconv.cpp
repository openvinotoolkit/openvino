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
    auto func_mod = [](long a) -> unsigned int { return a < 0 ? 0 : a; };
    auto pad_l = deconvAttrs.paddingL.size() > 1 ? deconvAttrs.paddingL.at(1) : deconvAttrs.paddingL.at(0);
    auto pad_r = deconvAttrs.paddingR.size() > 1 ? deconvAttrs.paddingR.at(1) : deconvAttrs.paddingR.at(0);
    auto pad_t = deconvAttrs.paddingL.at(0);
    auto pad_b = deconvAttrs.paddingR.at(0);

    unsigned int stride_x = (deconvAttrs.stride.size() > 1) ? deconvAttrs.stride.at(1) : deconvAttrs.stride.at(0);
    unsigned int stride_y = deconvAttrs.stride.at(0);
    auto deconv_info = PadStrideInfo(stride_x, stride_y, func_mod(pad_l), func_mod(pad_r), func_mod(pad_t), func_mod(pad_b), DimensionRoundingType::FLOOR);

    auto srcDims  = srcDescs[0]->getShape().getDims();
    auto weiDims  = srcDescs[1]->getShape().getDims();
    auto dstDims  = dstDescs[0]->getShape().getDims();

    // ACL can't work with custom output shape, this we make WA for that problem
    if (pad_l < 0 || pad_r < 0 || pad_t < 0 || pad_b < 0) {
        auto out_dims = deconvolution_output_dimensions(srcDims[3], srcDims[2], weiDims[3], weiDims[2], deconv_info);
        stride_x += (out_dims.first - dstDims[3] - 2 * (pad_l + pad_r)) / (srcDims[3] - 1);
        stride_y += (out_dims.second - dstDims[2] - 2 * (pad_t + pad_b)) / (srcDims[2] - 1);
        deconv_info = PadStrideInfo(stride_x, stride_y, func_mod(pad_l), func_mod(pad_r), func_mod(pad_t), func_mod(pad_b), DimensionRoundingType::FLOOR);
    }

    std::swap(weiDims[0], weiDims[1]);
    arm_compute::TensorShape srcVecDims = shapeCast(srcDims);
    arm_compute::TensorShape weiVecDims = shapeCast(weiDims);
    arm_compute::TensorShape dstVecDims = shapeCast(dstDims);
    arm_compute::TensorShape biasVecDims;
    if (deconvAttrs.withBiasesParam) {
        biasVecDims = shapeCast(srcDescs[2]->getShape().getDims());
    }
    if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        if (deconvAttrs.withBiasesParam) {
            changeLayoutToNH_C({&srcVecDims, &weiVecDims, &dstVecDims, &biasVecDims});
        } else {
            changeLayoutToNH_C({&srcVecDims, &weiVecDims, &dstVecDims});
        }
    }

    auto srcLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto weiLayout = getAclDataLayoutByMemoryDesc(srcDescs[1]);
    auto dstLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);

    if (srcLayout == arm_compute::DataLayout::NHWC && weiLayout == arm_compute::DataLayout::NCHW) {
        weiLayout = arm_compute::DataLayout::NHWC;
    }

    TensorInfo srcTensorInfo = TensorInfo(srcVecDims, 1,
                                          precisionToAclDataType(srcDescs[0]->getPrecision()), srcLayout);
    TensorInfo weiTensorInfo = TensorInfo(weiVecDims, 1,
                                          precisionToAclDataType(srcDescs[1]->getPrecision()), weiLayout);
    TensorInfo dstTensorInfo = TensorInfo(dstVecDims, 1,
                                          precisionToAclDataType(dstDescs[0]->getPrecision()), dstLayout);
    TensorInfo biasTensorInfo;
    if (deconvAttrs.withBiasesParam) {
        biasTensorInfo = TensorInfo(biasVecDims, 1,
                                    precisionToAclDataType(srcDescs[2]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[2]));
    }

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
    if (!AclDeconvExecutorBuilder::customIsSupported(deconvAttrs, srcDescs, dstDescs)) {
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);
    if (deconvAttrs.withBiasesParam)
        biasTensor.allocator()->init(biasTensorInfo);

    deconv = std::make_unique<arm_compute::NEDeconvolutionLayer>();
    configureThreadSafe([&] {
        deconv->configure(&srcTensor, &weiTensor, deconvAttrs.withBiasesParam ? &biasTensor : nullptr, &dstTensor, deconv_info, deconvAttrs.aclFastMath);
    });
    return true;
}

template<typename T>
static void transpose_weights(const MemoryCPtr& srcMemPtr, MemoryPtr& newSrcMemPtr, bool isNCHW) {
    const auto src_data = srcMemPtr->getDataAs<T>();
    const auto new_src_data = newSrcMemPtr->getDataAs<T>();

    const int DIM0 = static_cast<int>(srcMemPtr->getStaticDims()[0]);
    const int DIM1 = static_cast<int>(srcMemPtr->getStaticDims()[1]);
    const int DIM2 = static_cast<int>(srcMemPtr->getStaticDims()[2]);
    const int DIM3 = static_cast<int>(srcMemPtr->getStaticDims()[3]);

    // 0123 -> 1023
    if (isNCHW) {
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
                new_src_data[dst_off] = src_data[src_off];
            }
        });
    // 0231 -> 1230
    } else {
        parallel_for3d(DIM0, DIM1, DIM2, [&](const int dim0, const int dim1, const int dim2) {
            for (int dim3 = 0; dim3 < DIM3; ++dim3) {
                const int src_off = dim0 * DIM1 * DIM2 * DIM3 +
                                    dim1 * DIM2 * DIM3 +
                                    dim2 * DIM3 +
                                    dim3;
                const int dst_off = dim1 * DIM2 * DIM3 * DIM0 +
                                    dim2 * DIM3 * DIM0 +
                                    dim3 * DIM0 +
                                    dim0;
                new_src_data[dst_off] = src_data[src_off];
            }
        });
    }
}

static MemoryPtr prepareWeightMemory(const std::vector<MemoryCPtr>& src, const ExecutorContext::CPtr context) {
    DEBUG_LOG("ACLDeconvExecutor: prepack weights");
    const auto C = src[1]->getStaticDims()[1];
    const auto N = src[1]->getStaticDims()[0];

    auto create = [&]() {
        MemoryPtr newWei = std::make_shared<Memory>(context->getEngine(), src[1]->getDesc());
        if (src[0]->getDescPtr()->getPrecision() == element::Type_t::f16) {
            transpose_weights<ov::float16>(src[1], newWei, src[0]->getDescPtr()->hasLayoutType(LayoutType::ncsp));
        }
        if (src[0]->getDescPtr()->getPrecision() == element::Type_t::f32) {
            transpose_weights<float>(src[1], newWei, src[0]->getDescPtr()->hasLayoutType(LayoutType::ncsp));
        }
        return newWei;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        std::string format = "deconv_acl_" + std::to_string(N) + "_" + std::to_string(C);
        const std::string string_hash = format + "_" + std::to_string(src[1]->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(src[1]->getData()));
        DEBUG_LOG("ACLDeconvExecutor: findOrCreate, string_hash: ", string_hash);
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("ACLDeconvExecutor: Weights cache is not available");
    return create();
}

void AclDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    // TODO: Remove transpose from exec
    auto newWei = prepareWeightMemory(src, context);

    srcTensor.allocator()->import_memory(src[0]->getData());
    weiTensor.allocator()->import_memory(newWei->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());
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

    if (!(one_of(srcDescs[0]->getPrecision(), ov::element::f16, ov::element::f32) &&
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
          // Check weights as ncsp because we remove reorder and will transform ncsp -> nspc in exec() function
          srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
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
    auto srcTensorInfo  = aclDeconvTensorInfo.srcTensorInfo;
    auto weiTensorInfo  = aclDeconvTensorInfo.weiTensorInfo;
    auto biasTensorInfo = aclDeconvTensorInfo.biasTensorInfo;
    auto dstTensorInfo  = aclDeconvTensorInfo.dstTensorInfo;
    auto deconv_info    = aclDeconvTensorInfo.deconv_info;

    // After stride=8 up-sampling in ACL Deconvolution layer slower than reference
    if (deconv_info.stride().first >= 8 || deconv_info.stride().second >= 8) {
        DEBUG_LOG("AclDeconvExecutor does not support strides > 8:");
        return false;
    }

    unsigned int dilation_x = (deconvAttrs.dilation.size() > 1) ? deconvAttrs.dilation.at(1) : deconvAttrs.dilation.at(0);
    unsigned int dilation_y = deconvAttrs.dilation.at(0);
    if (!one_of(dilation_x, static_cast<unsigned int >(0), static_cast<unsigned int >(1)) ||
        !one_of(dilation_y, static_cast<unsigned int >(0), static_cast<unsigned int >(1))) return false;

    try {
        arm_compute::Status status = arm_compute::NEDeconvolutionLayer::validate(&srcTensorInfo,
                                                                                 &weiTensorInfo,
                                                                                 deconvAttrs.withBiasesParam ? &biasTensorInfo : nullptr,
                                                                                 &dstTensorInfo,
                                                                                 deconv_info,
                                                                                 deconvAttrs.aclFastMath);
        if (!status) {
            DEBUG_LOG("NEDeconvolutionLayer validation failed: ", status.error_description());
            return false;
        }
    } catch (...) {
        // Catch for ACL issue: https://github.com/ARM-software/ComputeLibrary/issues/1061
        DEBUG_LOG("NEDeconvolutionLayer validation failed with exception");
        return false;
    }

    return true;
}

}   // namespace intel_cpu
}   // namespace ov
