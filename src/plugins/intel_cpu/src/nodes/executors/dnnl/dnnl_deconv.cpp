// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_deconv.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <oneapi/dnnl/dnnl.hpp>

namespace ov {
namespace intel_cpu {

using DefaultDeconvDescs = std::pair<dnnl::convolution_backward_data::primitive_desc,
        dnnl::convolution_forward::primitive_desc>;

namespace {
DefaultDeconvDescs createDescriptorInternalDefault(const dnnl::memory::desc& in_candidate,
                                                   const dnnl::memory::desc& wgh_candidate,
                                                   const dnnl::memory::desc& out_candidate,
                                                   const dnnl::algorithm alg,
                                                   const std::vector<ptrdiff_t>& stride,
                                                   const std::vector<ptrdiff_t>& dilation,
                                                   const ov::CoordinateDiff& paddingL,
                                                   const ov::CoordinateDiff& paddingR,
                                                   const dnnl::primitive_attr& attr,
                                                   const dnnl::engine& engine) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return dnnl::memory::dims(orig_dims.begin(), orig_dims.end());
    };

    const dnnl::primitive_attr emptyAttr;

    auto conv_desc = dnnl::convolution_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            alg,
            out_candidate, wgh_candidate, in_candidate,
            convertDims(stride),
            convertDims(dilation),
            convertDims(paddingL),
            convertDims(paddingR),
            emptyAttr,
            true);

    if (!conv_desc.get(true)) {
        return {nullptr, nullptr};
    }

    auto deconv_desc = dnnl::convolution_backward_data::primitive_desc(
            engine,
            alg,
            out_candidate, wgh_candidate, in_candidate,
            convertDims(stride),
            convertDims(dilation),
            convertDims(paddingL),
            convertDims(paddingR),
            conv_desc,
            attr,
            true);

    return {deconv_desc, conv_desc};
}

dnnl::primitive_desc createDescriptorInternalInt8(const dnnl::memory::desc& in_candidate,
                                                  const dnnl::memory::desc& wgh_candidate,
                                                  const dnnl::memory::desc& bias_candidate,
                                                  const dnnl::memory::desc& out_candidate,
                                                  const bool with_bias,
                                                  const std::vector<ptrdiff_t>& stride,
                                                  const std::vector<ptrdiff_t>& dilation,
                                                  const ov::CoordinateDiff& paddingL,
                                                  const ov::CoordinateDiff& paddingR,
                                                  const dnnl::primitive_attr& attr,
                                                  const dnnl::engine& engine) {
    auto convertDims = [] (const std::vector<ptrdiff_t>& orig_dims) {
        return dnnl::memory::dims(orig_dims.begin(), orig_dims.end());
    };

    if (with_bias) {
        return dnnl::deconvolution_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::deconvolution_direct,
                in_candidate, wgh_candidate, bias_candidate, out_candidate,
                convertDims(stride), convertDims(dilation),
                convertDims(paddingL), convertDims(paddingR),
                attr);
    } else {
        return dnnl::deconvolution_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::deconvolution_direct,
                in_candidate, wgh_candidate, out_candidate,
                convertDims(stride), convertDims(dilation),
                convertDims(paddingL), convertDims(paddingR),
                attr);
    }
}

DefaultDeconvDescs createDefaultMkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                 const dnnl::memory::desc& wghDesc,
                                                 const dnnl::memory::desc& dstDesc,
                                                 bool isWinograd,
                                                 const std::vector<ptrdiff_t>& stride,
                                                 const std::vector<ptrdiff_t>& dilation,
                                                 const ov::CoordinateDiff& paddingL,
                                                 const ov::CoordinateDiff& paddingR,
                                                 const dnnl::primitive_attr& attr,
                                                 const dnnl::engine& engine) {
    dnnl::algorithm alg = isWinograd ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct;
    dnnl::convolution_backward_data::primitive_desc deconv_desc;
    dnnl::convolution_forward::primitive_desc fwd_conv_pd;
    std::tie(deconv_desc, fwd_conv_pd) = createDescriptorInternalDefault(srcDesc, wghDesc, dstDesc, alg, stride, dilation, paddingL, paddingR, attr, engine);
    if (fwd_conv_pd.get(true) == nullptr) {
        IE_THROW() << "Forward convolution primitive descriptor is nullable";
    }

    return {deconv_desc, fwd_conv_pd};
}

dnnl::primitive_desc createInt8MkldnnDeconvDesc(const dnnl::memory::desc& srcDesc,
                                                const dnnl::memory::desc& wghDesc,
                                                const dnnl::memory::desc& biasDesc,
                                                const dnnl::memory::desc& dstDesc,
                                                const bool withBias,
                                                const std::vector<ptrdiff_t>& stride,
                                                const std::vector<ptrdiff_t>& dilation,
                                                const ov::CoordinateDiff& paddingL,
                                                const ov::CoordinateDiff& paddingR,
                                                const dnnl::primitive_attr& attr,
                                                const dnnl::engine& engine) {
    return createDescriptorInternalInt8(
            srcDesc, wghDesc, biasDesc, dstDesc, withBias, stride, dilation, paddingL, paddingR, attr, engine);
}
} // namespace

//FIXME: add context
DNNLDeconvExecutor::DNNLDeconvExecutor() : DeconvExecutor() {}

bool DNNLDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    auto builder = [&deconvAttrs](const DeconvKey& key) -> executorPtr {
        dnnl::primitive_desc desc;
        dnnl::convolution_forward::primitive_desc fwd_conv_pd;
        dnnl::memory::desc dnnlBiasDesc;
        if (key.isInt8) {
            if (key.bias)
                dnnlBiasDesc = key.bias->getDnnlDesc();

            desc = createInt8MkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), dnnlBiasDesc, key.out->getDnnlDesc(),
                                              key.bias != nullptr, key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
        } else {
            std::tie(desc, fwd_conv_pd) = createDefaultMkldnnDeconvDesc(key.inp0->getDnnlDesc(), key.inp1->getDnnlDesc(), key.out->getDnnlDesc(),
                                                                        (key.implType & impl_desc_type::winograd),
                                                                        key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
        }

        dnnl::primitive_desc_iterator itpd = desc;
        executorPtr execPtr = nullptr;

        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            if (impl_type == key.implType) {
                if (key.isInt8) {
                    auto prim_desc = dnnl::deconvolution_forward::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
                                                                   key.inp0->getDnnlDesc(),
                                                                   key.inp1->getDnnlDesc(),
                                                                   key.out->getDnnlDesc(),
                                                                   deconvAttrs.engine);
                } else {
                    auto prim_desc = dnnl::convolution_backward_data::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
                                                                      key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      deconvAttrs.engine);
                }
                break;
            }

            if (!itpd.next_impl()) {
                break;
            }
        }

        if (!execPtr) {
            auto inDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp0->getShape().getStaticDims()),
                                             key.inp0->getDataType(),
                                             dnnl::memory::format_tag::any);
            auto wghDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.inp1->getShape().getStaticDims()),
                                              key.inp1->getDataType(),
                                              dnnl::memory::format_tag::any);
            auto outDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(key.out->getShape().getStaticDims()),
                                              key.out->getDataType(),
                                              dnnl::memory::format_tag::any);

            dnnl::primitive_desc anyDeconvDesc;
            dnnl::convolution_forward::primitive_desc fwdConvPd;

            if (key.isInt8) {
                anyDeconvDesc = createInt8MkldnnDeconvDesc(inDesc, wghDesc, dnnlBiasDesc, outDesc, key.bias != nullptr,
                                                           key.stride, key.dilation, key.paddingL, key.paddingR, key.attr, deconvAttrs.engine);
            } else {
                std::tie(anyDeconvDesc, fwdConvPd) = createDefaultMkldnnDeconvDesc(inDesc, wghDesc, outDesc, (key.implType & impl_desc_type::winograd),
                                                                                   key.stride, key.dilation, key.paddingL, key.paddingR, key.attr,
                                                                                   deconvAttrs.engine);
            }

            auto anyDeconvItpd = anyDeconvDesc;

            if (anyDeconvItpd) {
                if (key.isInt8) {
                    auto prim_desc = dnnl::deconvolution_forward::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorInt8>(prim_desc,
                                                                   key.inp0->getDnnlDesc(),
                                                                   key.inp1->getDnnlDesc(),
                                                                   key.out->getDnnlDesc(),
                                                                   deconvAttrs.engine);
                } else {
                    auto prim_desc = dnnl::convolution_backward_data::primitive_desc(itpd.get());
                    execPtr = std::make_shared<DeconvExecutorDefault>(prim_desc,
                                                                      key.inp0->getDnnlDesc(),
                                                                      key.inp1->getDnnlDesc(),
                                                                      key.out->getDnnlDesc(),
                                                                      deconvAttrs.engine);
                }
            }
        }

        return execPtr;
    };

    execPtr = nullptr;
    auto result = deconvAttrs.cache->getOrCreate(deconvAttrs.key, builder);

    execPtr = result.first;
    return true;
}

void DNNLDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
}

DNNLDeconvExecutor::DeconvExecutorDefault::DeconvExecutorDefault(const dnnl::convolution_backward_data::primitive_desc& pd,
                                                                 const dnnl::memory::desc& inMemDesc,
                                                                 const dnnl::memory::desc& weightMemDesc,
                                                                 const dnnl::memory::desc& outMemDesc,
                                                                 const dnnl::engine& engine) : DnnlExecutor(pd) {
    if (inMemDesc != pd.diff_dst_desc()) {
        inputReorders.insert({DNNL_ARG_DIFF_DST, IntermReorder(inMemDesc, pd.diff_dst_desc(), engine)});
    }

    if (weightMemDesc != pd.weights_desc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, pd.weights_desc(), engine)});
    }

    if (outMemDesc != pd.diff_src_desc()) {
        outputReorders.insert({DNNL_ARG_DIFF_SRC, IntermReorder(pd.diff_src_desc(), outMemDesc, engine)});
    }
}

DNNLDeconvExecutor::DeconvExecutorInt8::DeconvExecutorInt8(const dnnl::deconvolution_forward::primitive_desc &pd,
                                                           const dnnl::memory::desc &inMemDesc,
                                                           const dnnl::memory::desc &weightMemDesc,
                                                           const dnnl::memory::desc &outMemDesc,
                                                           const dnnl::engine &engine) : DnnlExecutor(pd) {
    if (inMemDesc != getDnnlSrcDesc()) {
        inputReorders.insert({DNNL_ARG_SRC, IntermReorder(inMemDesc, getDnnlSrcDesc(), engine)});
    }

    if (weightMemDesc != getDnnlWeightDesc()) {
        inputReorders.insert({DNNL_ARG_WEIGHTS, IntermReorder(weightMemDesc, getDnnlWeightDesc(), engine)});
    }

    if (outMemDesc != getDnnlDstDesc()) {
        outputReorders.insert({DNNL_ARG_DST, IntermReorder(getDnnlDstDesc(), outMemDesc, engine)});
    }
}
}   // namespace intel_cpu
}   // namespace ov