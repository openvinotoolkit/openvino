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

impl_desc_type DnnlExecutor2::getImplementationType() const {
    auto pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}

void DnnlExecutor2::setArg(int arg_id, dnnl::memory external_mem, bool isConst) {
    memory::desc internal_desc;
    bool is_output = false;

    if (arg_id >= DNNL_ARG_SRC && arg_id <= DNNL_ARG_SRC_3) {
        internal_desc = pd.src_desc(arg_id - DNNL_ARG_SRC);
    } else if (arg_id >= DNNL_ARG_WEIGHTS && arg_id <= DNNL_ARG_WEIGHTS_3) {
        internal_desc = pd.weights_desc(arg_id - DNNL_ARG_WEIGHTS);
    } else if (arg_id >= DNNL_ARG_DST && arg_id <= DNNL_ARG_DST_2) {
        internal_desc = pd.dst_desc(arg_id - DNNL_ARG_DST);
        is_output = true;
    } else {
        // normal ARGS w/o reorder
        args[arg_id] = external_mem;
        return;
    }

    // handl reorders for SRC/WEIGHTS/DST
    auto external_desc = external_mem.get_desc();
    if (external_desc == internal_desc) {
        args[arg_id] = external_mem;
    } else {
        dnnl::memory internal_mem;
        if (isConst) {
            internal_mem = addConstFolding(external_mem,
                                       std::to_string(arg_id),
                                       DnnlExtensionUtils::makeDescriptor(internal_desc));
        } else {
            internal_mem = dnnl::memory(internal_desc, context->getEngine());
            if (is_output) {
                outputReorders.emplace_back(context->getReorderPrim(internal_desc, external_desc),
                                            PrimArgs{{DNNL_ARG_SRC, internal_mem}, {DNNL_ARG_DST, external_mem}});
            } else {
                inputReorders.emplace_back(context->getReorderPrim(external_desc, internal_desc),
                                           PrimArgs{{DNNL_ARG_SRC, external_mem}, {DNNL_ARG_DST, internal_mem}});
            }
        }
        args[arg_id] = internal_mem;
    }
}

void DnnlExecutor2::setScratchPad() {
    if (pd.get_primitive_attr().get_scratchpad_mode() == dnnl::scratchpad_mode::user) {
        auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(pd.scratchpad_desc());
        scratchpadMem = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);
        args[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
    } else {
        scratchpadMem = nullptr;
    }
}

dnnl::memory DnnlExecutor2::addConstFolding(dnnl::memory src,
                                            std::string privateSrcKey,
                                            DnnlMemoryDescPtr expectedWeightDesc) {
    ConstFolding cf;
    cf.src_mem = src;
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
    auto srcWeightDesc = cf.src_mem.get_desc();
    if (srcWeightDesc.data.format_kind == dnnl::impl::format_kind::blocked) {
        // in fullyconnect layer, src weight's shape may be different but compatible with expected weight layout
        srcWeightDesc = srcWeightDesc.reshape(expectedDesc.dims());
    }

    auto create = [&]() {
        auto newSrcDesc = DnnlExtensionUtils::makeDescriptor(srcWeightDesc);
        Memory srcMemory{context->getEngine()};
        srcMemory.Create(newSrcDesc, cf.src_mem.get_data_handle());
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine());
        _ptr->Create(DnnlExtensionUtils::makeDescriptor(expectedDesc));
        context->reorderData(srcMemory, *_ptr);
        DEBUG_LOG("ConstFolding ", srcWeightDesc, " -> ", expectedDesc);
        return _ptr;
    };

    MemoryPtr ptr;
    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const std::string unique_name = cf.key + "_" + std::to_string(cf.src_mem.get_desc().get_size()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(cf.src_mem.get_data_handle()));
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
