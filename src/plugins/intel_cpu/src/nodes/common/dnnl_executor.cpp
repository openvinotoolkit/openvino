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

dnnl::memory::desc DnnlExecutor2::queryMD(const dnnl::query& what, int idx) {
    const dnnl_memory_desc_t* cdesc = dnnl_primitive_desc_query_md(pd, dnnl::convert_to_c(what), idx);
    return dnnl::memory::desc(*cdesc);
}

impl_desc_type DnnlExecutor2::getImplementationType() const {
    auto pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}

void DnnlExecutor2::setSrc(MemoryPtr memPtr, bool isConst, int idx) {
    dnnl::memory ext_mem = memPtr->GetPrimitive();
    auto ext_desc = ext_mem.get_desc();
    auto prim_desc = queryMD(dnnl::query::src_md, idx);
    if (ext_desc == prim_desc) {
        args[DNNL_ARG_SRC + idx] = ext_mem;
    } else {
        dnnl::memory prim_mem;
        if (isConst) {
            prim_mem = addConstFolding(memPtr, "s_" + std::to_string(idx), DnnlExtensionUtils::makeDescriptor(prim_desc));
        } else {
            prim_mem = dnnl::memory(prim_desc, context->getEngine());
            inputReorders.emplace_back(context->getReorderPrim(ext_desc, prim_desc),
                                        PrimArgs{{DNNL_ARG_SRC, ext_mem}, {DNNL_ARG_DST, prim_mem}});
        }
        args[DNNL_ARG_SRC + idx] = prim_mem;
    }
}

void DnnlExecutor2::setWeight(MemoryPtr memPtr, bool isConst, int idx) {
    dnnl::memory ext_mem = memPtr->GetPrimitive();
    auto ext_desc = ext_mem.get_desc();
    auto prim_desc = queryMD(dnnl::query::weights_md, idx);
    if (ext_desc == prim_desc) {
        args[DNNL_ARG_WEIGHTS + idx] = ext_mem;
    } else {
        dnnl::memory prim_mem;
        if (isConst) {
            // reordering of weight still need to be done at first execution, weight_mem is a stub with valid desc but no handle
            prim_mem = addConstFolding(memPtr, "w_" + std::to_string(idx), DnnlExtensionUtils::makeDescriptor(prim_desc));
        } else {
            // reordering of weight has to be done at execution time
            prim_mem = dnnl::memory(prim_desc, context->getEngine());
            inputReorders.emplace_back(context->getReorderPrim(ext_desc, prim_desc),
                                        PrimArgs{{DNNL_ARG_SRC, ext_mem}, {DNNL_ARG_DST, prim_mem}});
        }
        args[DNNL_ARG_WEIGHTS + idx] = prim_mem;
    }
}

void DnnlExecutor2::setOutput(MemoryPtr memPtr, int idx) {
    dnnl::memory ext_mem = memPtr->GetPrimitive();
    auto ext_desc = ext_mem.get_desc();
    auto prim_desc = queryMD(dnnl::query::dst_md, idx);
    if (ext_desc == prim_desc) {
        // directly output to final location
        args[DNNL_ARG_DST + idx] = ext_mem;
    } else {
        // generate output reorder after conv
        dnnl::memory prim_mem = dnnl::memory(prim_desc, context->getEngine());
        outputReorders.emplace_back(context->getReorderPrim(prim_desc, ext_desc),
                                    PrimArgs{{DNNL_ARG_SRC, prim_mem}, {DNNL_ARG_DST, ext_mem}});
        args[DNNL_ARG_DST + idx] = prim_mem;
    }
}

void DnnlExecutor2::setScratchPad() {
    auto pd = getPrimitiveDesc();
    auto scratchpadMemoryDesc = DnnlExtensionUtils::query_md(pd, dnnl::query::scratchpad_md);
    scratchpadMem = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);
    args[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
}

dnnl::memory DnnlExecutor2::addConstFolding(MemoryPtr src,
                                            std::string privateSrcKey,
                                            DnnlMemoryDescPtr expectedWeightDesc) {
    ConstFolding cf;
    cf.src = src;
    cf.dst_mem = dnnl::memory(expectedWeightDesc->getDnnlDesc(), context->getEngine(), nullptr);
    cf.key = privateSrcKey + "_" + expectedWeightDesc->serializeFormat();
    constFoldings.push_back(cf);
    return cf.dst_mem;
}

void DnnlExecutor2::doConstFolding(ConstFolding& cf) {
    // try find it in private weight cache
    auto it = privateWeightCache.find(cf.key);
    if (privateWeightCache.end() != it) {
        cf.dst_mem.set_data_handle(it->second->GetData());
        return;
    }

    // create or get from weight cache
    const auto& expectedDesc = cf.dst_mem.get_desc();
    auto srcWeightDesc = cf.src->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
    if (srcWeightDesc.data.format_kind == dnnl::impl::format_kind::blocked) {
        // in fullyconnect layer, src weight's shape may be different but compatible with expected weight layout
        srcWeightDesc = srcWeightDesc.reshape(expectedDesc.dims());
    }

    auto create = [&]() {
        auto newSrcDesc = DnnlExtensionUtils::makeDescriptor(srcWeightDesc);
        Memory srcMemory{context->getEngine()};
        srcMemory.Create(newSrcDesc, cf.src->GetData());
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine());
        _ptr->Create(DnnlExtensionUtils::makeDescriptor(expectedDesc));
        context->reorderData(srcMemory, *_ptr);
        DEBUG_LOG("ConstFolding ", srcWeightDesc, " -> ", expectedDesc);
        return _ptr;
    };

    MemoryPtr ptr;
    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const std::string unique_name = cf.key + "_" + std::to_string(cf.src->GetSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(cf.src->GetData()));
        ptr = *weightCache->findOrCreate(unique_name, create);
    } else {
        ptr = create();
    }
    cf.dst_mem.set_data_handle(ptr->GetData());
    privateWeightCache[cf.key] = ptr;
}

void DnnlExecutor2::exec(dnnl::stream strm) {
    // const folding is done once at first execution
    if (!constFolded) {
        for (auto & cf : constFoldings)
            doConstFolding(cf);
        constFolded = true;
    }

    // pre-reordering
    for (auto & s : inputReorders) {
        s.prim.execute(strm, s.args);
    }

    prim.execute(strm, args);

    // post-reordering
    for (auto & s : outputReorders) {
        s.prim.execute(strm, s.args);
    }
}

void DnnlExecutor2::setDynamicBatch(int newBatch) {
    if (!prim) {
        IE_THROW() << "Can't set dynamic batch for node: " << name << ", because executor is not compiled";
    }

    if (needReordering()) {
        IE_THROW() << "Can't execute node " << name << " with dynamic batch via executor with reorders";
    }

    auto setDynamicBatch = [this](int argType, int newBatch) {
        auto param = args.find(argType);
        if (param != args.end()) {
            auto oldMem = param->second;
            dnnl::memory::desc newMemDesc(oldMem.get_desc());
            newMemDesc.data.dims[0] = newBatch;
            newMemDesc.data.padded_dims[0] = newBatch;
            dnnl::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
            args.at(argType) = newMem;
        }
    };

    if (!args.empty()) {
        setDynamicBatch(DNNL_ARG_SRC, newBatch);
        setDynamicBatch(DNNL_ARG_DST, newBatch);
        setDynamicBatch(DNNL_ARG_DIFF_SRC, newBatch);
        setDynamicBatch(DNNL_ARG_DIFF_DST, newBatch);
    }
}

}  // namespace intel_cpu
}   // namespace ov
