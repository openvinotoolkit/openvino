// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_deconv.hpp"
#include "ie_parallel.hpp"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

//FIXME: add context
DNNLDeconvExecutor::DNNLDeconvExecutor() : DeconvExecutor() {}

bool DNNLDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
//    auto builder = [&deconvAttrs](const DeconvKey& key) -> executorPtr {
//        dnnl::primitive_desc desc;
//        dnnl::convolution_forward::primitive_desc fwd_conv_pd;
//        dnnl::memory::desc dnnlBiasDesc;
//        if (key.isInt8) {
//            if (key.bias)
//                dnnlBiasDesc = key.bias->getDnnlDesc();
//
//            desc = createInt8MkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), dnnlBiasDesc, key.out->getDnnlDesc(),
//                                              key.bias != nullptr, key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
//        } else {
//            std::tie(desc, fwd_conv_pd) = createDefaultMkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), key.out->getDnnlDesc(),
//                                                                        (key.implType & impl_desc_type::winograd),
//                                                                        key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
//        }
//
//        dnnl::primitive_desc_iterator itpd = desc;
//        executorPtr execPtr = nullptr;
//
//        while (static_cast<bool>(itpd)) {
//            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
//
//            if (impl_type == key.implType) {
//                if (key.isInt8) {
//                    auto prim_desc = dnnl::deconvolution_forward::primitive_desc(itpd.get());
//                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
//                                                                   key.inp0->getDnnlDesc(),
//                                                                   key.inp1->getDnnlDesc(),
//                                                                   key.out->getDnnlDesc(),
//                                                                   deconvAttrs.engine);
//                } else {
//                    auto prim_desc = dnnl::convolution_backward_data::primitive_desc(itpd.get());
//                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
//                                                                      key.inp0->getDnnlDesc(),
//                                                                      key.inp1->getDnnlDesc(),
//                                                                      key.out->getDnnlDesc(),
//                                                                      deconvAttrs.engine);
//                }
//                break;
//            }
//
//            if (!itpd.next_impl()) {
//                break;
//            }
//        }
//
//        if (!execPtr) {
//            auto inDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp0->getShape().getStaticDims()),
//                                             key.inp0->getDataType(),
//                                             memory::format_tag::any);
//            auto wghDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
//                                              key.inp1->getDataType(),
//                                              memory::format_tag::any);
//            auto outDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
//                                              key.out->getDataType(),
//                                              memory::format_tag::any);
//
//            dnnl::primitive_desc anyDeconvDesc;
//            dnnl::convolution_forward::primitive_desc fwdConvPd;
//
//            if (key.isInt8) {
//                anyDeconvDesc = createInt8MkldnnDeconvDesc(inDesc, wghDesc, dnnlBiasDesc, outDesc, key.bias != nullptr,
//                                                           key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
//            } else {
//                std::tie(anyDeconvDesc, fwdConvPd) = createDefaultMkldnnDeconvDesc(inDesc, wghDesc, outDesc, (key.implType & impl_desc_type::winograd),
//                                                                                   key.stride, key.dilation, key.paddingL, key.paddingR, key.attr,
//                                                                                   deconvAttrs.engine);
//            }
//
//            auto anyDeconvItpd = anyDeconvDesc;
//
//            if (anyDeconvItpd) {
//                if (key.isInt8) {
//                    auto prim_desc = dnnl::deconvolution_forward::primitive_desc(itpd.get());
//                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
//                                                                   key.inp0->getDnnlDesc(),
//                                                                   key.inp1->getDnnlDesc(),
//                                                                   key.out->getDnnlDesc(),
//                                                                   deconvAttrs.engine);
//                } else {
//                    auto prim_desc = dnnl::convolution_backward_data::primitive_desc(itpd.get());
//                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
//                                                                      key.inp0->getDnnlDesc(),
//                                                                      key.inp1->getDnnlDesc(),
//                                                                      key.out->getDnnlDesc(),
//                                                                      deconvAttrs.engine);
//                }
//            }
//        }
//
//        return execPtr;
//    };
//
//    execPtr = nullptr;
//    auto cache = context->getParamsCache();
//    auto result = cache->getOrCreate(deconvAttrs.key, builder);
//
//    execPtr = result.first;
//
//    if (execPtr) {
//        if (deconvAttrs.key.isInt8) {
//            deconvAttrs.primArgs[DNNL_ARG_SRC] = srcMemPtr->GetPrimitive();
//            deconvAttrs.primArgs[DNNL_ARG_WEIGHTS] = internalBlobMemory.front()->GetPrimitive();
//            deconvAttrs.primArgs[DNNL_ARG_DST]=  dstMemPtr->GetPrimitive();
//            if (withBiases)
//                deconvAttrs.primArgs[DNNL_ARG_BIAS] = biasMemPtr->GetPrimitive();
//        } else {
//            deconvAttrs.primArgs[DNNL_ARG_DIFF_DST] = srcMemPtr->GetPrimitive();
//            deconvAttrs.primArgs[DNNL_ARG_WEIGHTS] = wghMemPtr->GetPrimitive();
//            deconvAttrs.primArgs[DNNL_ARG_DIFF_SRC] = dstMemPtr->GetPrimitive();
//        }
//        Node::appendPostOpArgs(*pAttrLocal, deconvAttrs.primArgs, postOpsArgs);
//
//        auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
//        deconvAttrs.primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
//#ifdef CPU_DEBUG_CAPS
//        if (result.second == CacheEntryBase::LookUpStatus::Miss) {
//            auto pd = execPtr->getPrimitiveDesc();
//            DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
//        }
//#endif
//    } else {
//        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
//    }
    return true;
}

void DNNLDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
}

}   // namespace intel_cpu
}   // namespace ov