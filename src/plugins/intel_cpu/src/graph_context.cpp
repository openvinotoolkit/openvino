// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <dnnl_types.h>
#include "graph_context.h"

#include <common/primitive_hashing_utils.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"

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

}   // namespace intel_cpu
}   // namespace ov
