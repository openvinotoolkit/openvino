// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_executor.h"

#include "utils/debug_capabilities.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {

DnnlExecutor::IntermReorder::IntermReorder(const dnnl::memory::desc& descSrc,
                                           const dnnl::memory::desc& descDst,
                                           const dnnl::engine& engine) : m_descSrc(descSrc), m_descDst(descDst) {
    auto reorderPd = dnnl::reorder::primitive_desc(engine, descSrc, engine, descDst);
    m_reorder = dnnl::reorder(reorderPd);
}

void DnnlExecutor::IntermReorder::exec(dnnl::memory& memSrc, dnnl::memory& memDst, dnnl::stream strm) {
    m_reorder.execute(strm, memSrc, memDst);
}

void DnnlExecutor::exec(std::unordered_map<int, dnnl::memory> primArgs, dnnl::stream strm) {
    for (auto &inReorder : inputReorders) {
        if (primArgs.count(inReorder.first)) {
            dnnl::memory memDst(inReorder.second.getDstDesc(), strm.get_engine());
            inReorder.second.exec(primArgs[inReorder.first], memDst, strm);
            primArgs[inReorder.first] = memDst;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for input " << inReorder.first << ", but doesn't have source memory";
        }
    }
    std::unordered_map<int, dnnl::memory> outputMem;
    for (auto &outReorder : outputReorders) {
        if (primArgs.count(outReorder.first)) {
            dnnl::memory memSrc(outReorder.second.getSrcDesc(), strm.get_engine());
            outputMem[outReorder.first] = primArgs[outReorder.first];
            primArgs[outReorder.first] = memSrc;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for output " << outReorder.first << ", but doesn't have destination memory";
        }
    }
    (*execPrim).execute(strm, primArgs);
    for (auto &outReorder : outputReorders) {
        outReorder.second.exec(primArgs[outReorder.first], outputMem[outReorder.first], strm);
    }
}

bool DnnlExecutor::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}

Primitive DnnlExecutor::getExecPrim() const {
    return execPrim;
}

const_dnnl_primitive_desc_t DnnlExecutor::getPrimitiveDesc() const {
    return (*execPrim).get_primitive_desc();
}

dnnl::memory::desc DnnlExecutor::getSrcDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::src_md);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor::getWeightDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::weights_md);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor::getDstDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::dst_md);

    return md->getDnnlDesc();
}

impl_desc_type DnnlExecutor::getImplementationType() const {
    auto pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}


bool DnnlExecutor2::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}

const_dnnl_primitive_desc_t DnnlExecutor2::getPrimitiveDesc() const {
    return prim.get_primitive_desc();
}

dnnl::memory::desc DnnlExecutor2::getSrcDesc(int idx) const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::src_md, idx);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor2::getWeightDesc(int idx) const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::weights_md, idx);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor2::getDstDesc(int idx) const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::dst_md);

    return md->getDnnlDesc();
}

impl_desc_type DnnlExecutor2::getImplementationType() const {
    auto pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}


void DnnlExecutor2::setSrc(MemoryPtr srcMemPtr, int idx) {
    dnnl::memory src_mem;
    auto node_src_desc = srcMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    auto prim_src_desc = getSrcDesc(idx);
    if (node_src_desc == prim_src_desc) {
        src_mem = srcMemPtr->GetPrimitive();
    } else {
        src_mem = dnnl::memory(prim_src_desc, context->getEngine());
        inputReorders.emplace_back(context->getReorderPrim(node_src_desc, prim_src_desc),
                                    PrimArgs{{DNNL_ARG_SRC, srcMemPtr->GetPrimitive()}, {DNNL_ARG_DST, src_mem}});
    }
    args[DNNL_ARG_SRC + idx] = src_mem;
}

void DnnlExecutor2::setWeight(MemoryPtr wghMemPtr, bool constWeight, int idx) {
    dnnl::memory weight_mem;
    auto node_weight_desc = wghMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    auto prim_weight_desc = getWeightDesc(0);
    if (node_weight_desc == prim_weight_desc) {
        weight_mem = wghMemPtr->GetPrimitive();
    } else {
        if (constWeight) {
            // reordering of weight can be done at this last compilation stage
            weight_mem =
                prepareWeightMemory(wghMemPtr, DnnlExtensionUtils::makeDescriptor(prim_weight_desc))->GetPrimitive();
        } else {
            // reordering of weight has to be done at execution time
            weight_mem = dnnl::memory(prim_weight_desc, context->getEngine());
            inputReorders.emplace_back(context->getReorderPrim(node_weight_desc, prim_weight_desc),
                                        PrimArgs{{DNNL_ARG_SRC, wghMemPtr->GetPrimitive()}, {DNNL_ARG_DST, weight_mem}});
        }
    }
    args[DNNL_ARG_WEIGHTS + idx] = weight_mem;
}

void DnnlExecutor2::setOutput(MemoryPtr dstMemPtr, int idx) {
    dnnl::memory dst_mem;
    auto node_dst_desc = dstMemPtr->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    auto prim_dst_desc = getDstDesc(idx);
    if (node_dst_desc == prim_dst_desc) {
        // directly output to final location
        dst_mem = dstMemPtr->GetPrimitive();
    } else {
        // generate output reorder after conv
        dst_mem = dnnl::memory(prim_dst_desc, context->getEngine());
        outputReorders.emplace_back(context->getReorderPrim(prim_dst_desc, node_dst_desc),
                                    PrimArgs{{DNNL_ARG_SRC, dst_mem}, {DNNL_ARG_DST, dstMemPtr->GetPrimitive()}});
    }
    args[DNNL_ARG_DST + idx] = dst_mem;
}


void DnnlExecutor2::setScratchPad() {
    auto pd = getPrimitiveDesc();
    auto scratchpadMemoryDesc = DnnlExtensionUtils::query_md(pd, dnnl::query::scratchpad_md);
    scratchpadMem = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);
    args[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
}

MemoryPtr DnnlExecutor2::prepareWeightMemory(MemoryPtr blob, DnnlMemoryDescPtr weightDesc) {
    const auto& format = weightDesc->serializeFormat();
    auto itr = privateWeightCache.find(format);
    if (privateWeightCache.end() != itr) {
        return itr->second;
    }

    auto constDnnlMemOutDesc = blob->GetDescWithType<DnnlMemoryDesc>();
    auto weightSrcDesc = constDnnlMemOutDesc->getDnnlDesc();
    weightSrcDesc = weightSrcDesc.reshape(weightDesc->getDnnlDesc().dims());
    auto create = [&] () {
        auto newSrcDesc = DnnlExtensionUtils::makeDescriptor(weightSrcDesc);

        Memory srcMemory{ context->getEngine() };
        srcMemory.Create(newSrcDesc, blob->GetData());

        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine());
        _ptr->Create(weightDesc);
        context->reorderData(srcMemory, *_ptr);

        DEBUG_LOG("prepareWeightMemory", *newSrcDesc, " -> ", *weightDesc);
        return _ptr;
    };

    MemoryPtr ptr;
    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const std::string string_hash = name + "_" + format
                                        + "_" + std::to_string(blob->GetSize())
                                        + "_" + std::to_string(reinterpret_cast<uint64_t>(blob->GetData()));

        ptr = *weightCache->findOrCreate(string_hash, create);
    } else {
        ptr = create();
    }
    privateWeightCache[format] = ptr;

    return ptr;
}

void DnnlExecutor2::exec(dnnl::stream strm) {
    for (auto & s : inputReorders) {
        s.prim.execute(strm, s.args);
    }
    prim.execute(strm, args);
    for (auto & s : outputReorders) {
        s.prim.execute(strm, s.args);
    }
}

}  // namespace intel_cpu
}   // namespace ov
