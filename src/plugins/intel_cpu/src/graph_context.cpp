// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <dnnl_types.h>
#include "graph_context.h"

#include <common/primitive_hashing_utils.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

dnnl::engine GraphContext::eng(dnnl::engine::kind::cpu, 0);

struct ReorderKey {
    dnnl::memory::desc src;
    dnnl::memory::desc dest;
    size_t hash() const;
    bool operator==(const ReorderKey& rhs) const;
};

size_t ReorderKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

dnnl::reorder GraphContext::getReorderPrim(const dnnl::memory::desc& src, const dnnl::memory::desc& dest) const {
    auto builder = [this](const ReorderKey& key) {
        dnnl::primitive_attr attr;
        //DEBUG_LOG(key.src, "->", key.dest);
        dnnl::reorder::primitive_desc pd = dnnl::reorder::primitive_desc(eng, key.src, eng, key.dest, attr, true);
        if (!pd) {
            return dnnl::reorder();
        }
        return dnnl::reorder(pd);
    };

    ReorderKey key = {src, dest};
    if (rtParamsCache) {
        auto result = rtParamsCache->getOrCreate(key, builder);
        return result.first;
    }
    return builder(key);
}

void GraphContext::reorderData(const Memory &input, const Memory &output) const {
    if (!input.getDesc().isDefined() || !output.getDesc().isDefined())
        IE_THROW() << "Can't reorder data with dynamic shapes";

    if (input.GetShape().hasZeroDims() || output.GetShape().hasZeroDims()) {
        return;
    }

    if (input.getDesc().isCompatible(output.getDesc())) {
        auto srcPtr = static_cast<uint8_t*>(input.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(output.GetPtr());

        auto copySize = output.GetSize();
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        dnnl::reorder pReorder;
        std::vector<uint8_t> tmpBuff;

        auto srcMemory = input.GetPrimitive();
        auto dstMemory = output.GetPrimitive();
        auto engine = output.getEngine();
        // try directly reorder
        pReorder = getReorderPrim(srcMemory.get_desc(), dstMemory.get_desc());
        if (!pReorder) {
            // try precision conversion then do the reorder
            //     bool isSupported = desc.getType() & MemoryDescType::Blocked;
            auto isSupportedDesc = [](const MemoryDesc& desc) {
                bool isSupported = desc.getType() & MemoryDescType::Blocked;
                if (desc.getType() == MemoryDescType::DnnlBlocked)
                    isSupported &= desc.as<const DnnlMemoryDesc>()->hasEmptyExtraData();
                return isSupported;
            };

            if (output.GetDataType() != input.GetDataType() && isSupportedDesc(input.getDesc()) &&
                isSupportedDesc(output.getDesc())) {
                // we probably could not make the reorder because there is no one supporting this precision conversion
                // lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t*>(input.GetPtr());
                tmpBuff.resize(input.GetSize());

                const auto outPrc = DnnlExtensionUtils::DataTypeToIEPrecision(output.GetDataType());
                cpu_convert(data,
                            tmpBuff.data(),
                            DnnlExtensionUtils::DataTypeToIEPrecision(input.GetDataType()),
                            outPrc,
                            input.GetSize() / input.getDesc().getPrecision().size());

                Memory tmpMem(engine);
                auto tmpDesc = input.getDesc().cloneWithNewPrecision(outPrc);
                tmpMem.Create(std::move(tmpDesc), tmpBuff.data());

                srcMemory = tmpMem.GetPrimitive();
                pReorder = getReorderPrim(srcMemory.get_desc(), dstMemory.get_desc());
            }
            if (!pReorder) {
                IE_THROW() << "No reorder available for the following tensor descriptors: "
                           << input.getDesc().serializeFormat() << " and " << output.getDesc().serializeFormat();
            }
        }
        if (pReorder) {
            dnnl::stream loc_stream(engine, dnnl::stream::flags::in_order);
            pReorder.execute(loc_stream, srcMemory, dstMemory);
        } else {
            IE_THROW() << "Could not make onednn reorder.";
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
struct DnnlOPKey {
    dnnl::memory::desc src;
    dnnl::memory::desc weight;
    dnnl::memory::desc bias;
    dnnl::memory::desc dst;
    dnnl::primitive_attr attr;
    impl_desc_type implType;
    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        for (const auto& ptr : {src, weight, bias, dst}) {
            if (ptr) {
                seed = hash_combine(seed, get_md_hash(ptr.data));
            }
        }
        seed = hash_combine(seed, get_attr_hash(*attr.get()));
        seed = hash_combine(seed, implType);
        return seed;
    }
    bool operator==(const DnnlOPKey& rhs) const {
        bool retVal = true;
        if (src != rhs.src) {
            retVal = retVal && src && rhs.src && src == rhs.src;
        }
        if (weight != rhs.weight) {
            retVal = retVal && rhs.weight && weight == rhs.weight;
        }
        if (bias != rhs.bias) {
            retVal = retVal && bias && rhs.bias && bias == rhs.bias;
        }
        if (dst != rhs.dst) {
            retVal = retVal && dst && rhs.dst && dst == rhs.dst;
        }
        retVal = retVal && *attr.get() == *rhs.attr.get() && implType == rhs.implType;
        return retVal;
    }
};
/////////////////////////////////////////////////////////////////////////////
struct ConvKey : public DnnlOPKey {
    using Prim = dnnl::convolution_forward;

    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;

    size_t hash() const;
    bool operator==(const ConvKey& rhs) const;

    dnnl::convolution_forward::desc createOPDescriptor() const {
        auto alg = (implType & impl_desc_type::winograd) ? dnnl::algorithm::convolution_winograd
                                                        : dnnl::algorithm::convolution_direct;
        if (bias) {
            // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
            auto biasNormalized = bias.reshape({dst.dims()[1]});

            return dnnl::convolution_forward::desc(dnnl::prop_kind::forward_scoring,
                                                   alg,
                                                   src,
                                                   weight,
                                                   biasNormalized,
                                                   dst,
                                                   dnnl::memory::dims(stride.begin(), stride.end()),
                                                   dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                   dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                   dnnl::memory::dims(paddingR.begin(), paddingR.end()));
        } else {
            return dnnl::convolution_forward::desc(dnnl::prop_kind::forward_scoring,
                                                   alg,
                                                   src,
                                                   weight,
                                                   dst,
                                                   dnnl::memory::dims(stride.begin(), stride.end()),
                                                   dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                   dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                   dnnl::memory::dims(paddingR.begin(), paddingR.end()));
        }
    }
};

size_t ConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = DnnlOPKey::hash();
    seed = get_vector_hash(seed, stride);
    seed = get_vector_hash(seed, dilation);
    seed = get_vector_hash(seed, paddingL);
    seed = get_vector_hash(seed, paddingR);
    return seed;
}

bool ConvKey::operator==(const ConvKey &rhs) const {
    bool retVal = DnnlOPKey::operator==(rhs);
    retVal = retVal && stride == rhs.stride;
    retVal = retVal && dilation == rhs.dilation;
    retVal = retVal && paddingL == rhs.paddingL;
    retVal = retVal && paddingR == rhs.paddingR;
    return retVal;
}

template<typename K>
std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getPrim(const K & key) const {
    auto builder = [this](const K & key) -> std::pair<dnnl::primitive, dnnl::primitive_desc_base> {
        auto op_desc = key.createOPDescriptor();
        auto itpd = dnnl::primitive_desc_iterator(&op_desc.data, &key.attr, getEngine(), nullptr, true);
        while (static_cast<bool>(itpd)) {
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (impl_type == key.implType || key.implType == impl_desc_type::undef) {
                auto prim_desc = typename K::Prim::primitive_desc(itpd.get());
                return std::make_pair(typename K::Prim(prim_desc), prim_desc);
            }
            if (!itpd.next_impl()) {
                break;
            }
        }
        // empty object indicating failure.
        DEBUG_LOG(" Failed for src=", key.src, " weight=", key.weight, " dst=", key.dst, " impl=", impl_type_to_string(key.implType));
        return std::make_pair(dnnl::primitive(), dnnl::primitive_desc());
    };

    auto result = rtParamsCache->getOrCreate(key, builder);

    return result.first;
}

std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getConvPrim(const dnnl::memory::desc &src,
                                                                                const dnnl::memory::desc &weight,
                                                                                const dnnl::memory::desc &bias,
                                                                                const dnnl::memory::desc &dst,
                                                                                const std::vector<ptrdiff_t> &stride,
                                                                                const std::vector<ptrdiff_t> &dilation,
                                                                                const std::vector<ptrdiff_t> &paddingL,
                                                                                const std::vector<ptrdiff_t> &paddingR,
                                                                                const dnnl::primitive_attr &attr,
                                                                                const impl_desc_type &implType) const {
    ConvKey key;
    key.src = src;
    key.weight = weight;
    key.bias = bias;
    key.dst = dst;
    key.stride = stride;
    key.dilation = dilation;
    key.paddingL = paddingL;
    key.paddingR = paddingR;
    key.attr = attr;
    key.implType = implType;

    return getPrim(key);
}


/////////////////////////////////////////////////////////////////////////////
struct InnerProductKey : public DnnlOPKey {
    using Prim = dnnl::inner_product_forward;

    dnnl::inner_product_forward::desc createOPDescriptor() const {
        if (bias) {
            return dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_scoring, src, weight, bias, dst);
        } else {
            return dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_scoring, src, weight, dst);
        }
    }
};

std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getInnerProductPrim(
    const dnnl::memory::desc& src,
    const dnnl::memory::desc& weight,
    const dnnl::memory::desc& bias,
    const dnnl::memory::desc& dst,
    const dnnl::primitive_attr& attr,
    const impl_desc_type& implType) const {
    InnerProductKey key;
    key.src = src;
    key.weight = weight;
    key.bias = bias;
    key.dst = dst;
    key.attr = attr;
    key.implType = implType;
    return getPrim(key);
}
/////////////////////////////////////////////////////////////////////////////
struct ConvBackwardDataKey : public ConvKey {
    using Prim = dnnl::convolution_backward_data;
    dnnl::convolution_backward_data::desc createOPDescriptor() const {
        auto alg = (implType & impl_desc_type::winograd) ? dnnl::algorithm::convolution_winograd
                                                         : dnnl::algorithm::convolution_direct;
        return dnnl::convolution_backward_data::desc(alg,
                                                     dst,   // diff_src_desc
                                                     weight,
                                                     src,   // diff_dst_desc
                                                     dnnl::memory::dims(stride.begin(), stride.end()),
                                                     dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                     dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                     dnnl::memory::dims(paddingR.begin(), paddingR.end()));
    }
};

std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getConvBackPrim(
    const dnnl::memory::desc& src,
    const dnnl::memory::desc& weight,
    const dnnl::memory::desc& dst,
    const std::vector<ptrdiff_t>& stride,
    const std::vector<ptrdiff_t>& dilation,
    const std::vector<ptrdiff_t>& paddingL,
    const std::vector<ptrdiff_t>& paddingR,
    const dnnl::primitive_attr& attr,
    const impl_desc_type& implType) const {
    ConvBackwardDataKey key;
    key.src = src;
    key.weight = weight;
    key.dst = dst;
    key.stride = stride;
    key.dilation = dilation;
    key.paddingL = paddingL;
    key.paddingR = paddingR;
    key.attr = attr;
    key.implType = implType;
    return getPrim(key);
}
/////////////////////////////////////////////////////////////////////////////
struct DeconvKey : public ConvKey {
    using Prim = dnnl::deconvolution_forward;
    dnnl::deconvolution_forward::desc createOPDescriptor() const {
        auto NormalizedBias = bias;
        if (NormalizedBias) {
            // WA to align IR bias representation (3 to 5 rank tensors) to oneDNN representation (1 rank tensor)
            NormalizedBias = NormalizedBias.reshape({static_cast<dnnl::memory::dim>(dst.dims()[1])});
        }
        return dnnl::deconvolution_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::deconvolution_direct,
                                                 src,
                                                 weight,
                                                 NormalizedBias,
                                                 dst,
                                                 dnnl::memory::dims(stride.begin(), stride.end()),
                                                 dnnl::memory::dims(dilation.begin(), dilation.end()),
                                                 dnnl::memory::dims(paddingL.begin(), paddingL.end()),
                                                 dnnl::memory::dims(paddingR.begin(), paddingR.end()));
    }
};

std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getDeconvPrim(
    const dnnl::memory::desc& src,
    const dnnl::memory::desc& weight,
    const dnnl::memory::desc& bias,
    const dnnl::memory::desc& dst,
    const std::vector<ptrdiff_t>& stride,
    const std::vector<ptrdiff_t>& dilation,
    const std::vector<ptrdiff_t>& paddingL,
    const std::vector<ptrdiff_t>& paddingR,
    const dnnl::primitive_attr& attr,
    const impl_desc_type& implType) const {
    DeconvKey key;
    key.src = src;
    key.weight = weight;
    key.bias = bias;
    key.dst = dst;
    key.stride = stride;
    key.dilation = dilation;
    key.paddingL = paddingL;
    key.paddingR = paddingR;
    key.attr = attr;
    key.implType = implType;
    return getPrim(key);
}

/////////////////////////////////////////////////////////////////////////////
struct MatMulKey : public DnnlOPKey {
    using Prim = dnnl::matmul;

    dnnl::matmul::desc createOPDescriptor() const {
        if (bias) {
            return dnnl::matmul::desc(src, weight, bias, dst);
        } else {
            return dnnl::matmul::desc(src, weight, dst);
        }
    }
};

std::pair<dnnl::primitive, dnnl::primitive_desc_base> GraphContext::getMatMulPrim(
    const dnnl::memory::desc& src,
    const dnnl::memory::desc& weight,
    const dnnl::memory::desc& bias,
    const dnnl::memory::desc& dst,
    const dnnl::primitive_attr& attr,
    const impl_desc_type& implType) const {
    MatMulKey key;
    key.src = src;
    key.weight = weight;
    key.bias = bias;
    key.dst = dst;
    key.attr = attr;
    key.implType = implType;
    return getPrim(key);
}

}   // namespace intel_cpu
}   // namespace ov
