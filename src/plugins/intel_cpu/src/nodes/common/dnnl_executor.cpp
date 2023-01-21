// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_executor.h"

#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>

#include "utils/debug_capabilities.h"
using namespace dnnl;

namespace ov {
namespace intel_cpu {

bool DnnlExecutor::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}

impl_desc_type DnnlExecutor::getImplementationType() const {
    return parse_impl_name(pd.impl_info_str());
}

void DnnlExecutor::reset(dnnl::primitive p, dnnl::primitive_desc_base prim_desc) {
    prim = p;
    pd = prim_desc;
    DEBUG_LOG("CPU node      : ", name);
    DEBUG_LOG("  primitive   : ", p.get_primitive_desc()->info());
    DEBUG_LOG("  src_desc    : ", pd.src_desc(0));
    DEBUG_LOG("  weights_desc: ", pd.weights_desc(0));
    DEBUG_LOG("  dst_desc    : ", pd.dst_desc(0));
    inputReorders.clear();
    outputReorders.clear();
    constFoldings.clear();
    canonicalizations.clear();
    constFolded = false;
    // privateWeightCache was kept to always reference weights of different format
    setScratchPad();
}
// scenario
//    infer[i-1](shape1) ===>  infer[i](shape2) ==> infer[i+1](shape2)
//
// shape change happens at infer[i], DnnlExecutor will be updated with
// the input/output/internal memory objects of new sizes.
//
// at infer[i+1], shape is unchanged, DnnlExecutor is not updated, so
//  1. descriptors of input/output/internal memory
//  2. the primitives: main or internal reorders
// are all unchanged, but the memory pointer may be updated, especially
// when it's from Input node, due to some optimizations for eliminating copies.
//
// because external_mem is referencing the same dnnl::memory object referenced by Edges,
// thus any changes to the actual addresses still can be sensed by all
// primitives in the DnnlExecutor.
//
// things becomes different when primitive requires canonicalized memory,
// at infer[i+1], the canonicalized memory object's handle needs to be updated
// explicitly, we maintain such record in updateHandles.
//
void DnnlExecutor::setArg(int arg_id,
                          dnnl::memory external_mem,
                          bool is_const,
                          const dnnl::memory::desc* p_canonical_desc) {
    memory::desc internal_desc;
    bool is_output = false;

    auto canonical2str = [&]() {
        std::stringstream ss;
#ifdef CPU_DEBUG_CAPS
        if (p_canonical_desc) {
            ss << " (canonicalized as " << (*p_canonical_desc) << ")";
        }
#endif
        return ss.str();
    };

    if (arg_id >= DNNL_ARG_SRC && arg_id <= DNNL_ARG_SRC_3) {
        internal_desc = pd.src_desc(arg_id - DNNL_ARG_SRC);
        DEBUG_LOG(" DNNL_ARG_SRC_", arg_id - DNNL_ARG_SRC, " is set to ", external_mem.get_desc(), canonical2str());
    } else if (arg_id >= DNNL_ARG_WEIGHTS && arg_id <= DNNL_ARG_WEIGHTS_3) {
        internal_desc = pd.weights_desc(arg_id - DNNL_ARG_WEIGHTS);
        DEBUG_LOG(" DNNL_ARG_WEIGHTS_",
                  arg_id - DNNL_ARG_WEIGHTS,
                  " is set to ",
                  external_mem.get_desc(),
                  canonical2str());
    } else if (arg_id >= DNNL_ARG_DST && arg_id <= DNNL_ARG_DST_2) {
        internal_desc = pd.dst_desc(arg_id - DNNL_ARG_DST);
        is_output = true;
        DEBUG_LOG(" DNNL_ARG_DST_", arg_id - DNNL_ARG_DST, " is set to ", external_mem.get_desc(), canonical2str());
    } else if (arg_id >= DNNL_ARG_DIFF_DST && arg_id <= DNNL_ARG_DIFF_DST_2) {
        // for convolution_backward_data, diff_dst is input
        internal_desc = pd.diff_dst_desc(arg_id - DNNL_ARG_DIFF_DST);
        DEBUG_LOG(" DNNL_ARG_DIFF_DST_",
                  arg_id - DNNL_ARG_DIFF_DST,
                  " is set to ",
                  external_mem.get_desc(),
                  canonical2str());
    } else if (arg_id >= DNNL_ARG_DIFF_SRC && arg_id <= DNNL_ARG_DIFF_SRC_3) {
        // for convolution_backward_data, diff_src is output
        internal_desc = pd.diff_src_desc(arg_id - DNNL_ARG_DIFF_SRC);
        is_output = true;
        DEBUG_LOG(" DNNL_ARG_DIFF_SRC_",
                  arg_id - DNNL_ARG_DIFF_SRC,
                  " is set to ",
                  external_mem.get_desc(),
                  canonical2str());
    } else {
        // normal ARGS w/o reorder
        args[arg_id] = external_mem;
        return;
    }

    if (is_const) {
        // runtime reordering is done in const-folding stage
        // and :
        //  - handle is retrieved from external_mem when folding
        //  - if provided, canonical desc will be used before reordering
        args[arg_id] = addConstFolding(external_mem,
                                       p_canonical_desc,
                                       std::to_string(arg_id),
                                       DnnlExtensionUtils::makeDescriptor(internal_desc));
        return;
    }

    auto external_desc = external_mem.get_desc();
    bool external_is_canonical = ((!p_canonical_desc) || (*p_canonical_desc == external_desc));
    if (external_is_canonical) {
        if (external_desc == internal_desc) {
            // matches internal desc, directly use it
            args[arg_id] = external_mem;
        } else {
            // directly reorder from/to external_mem
            dnnl::memory internal_mem = dnnl::memory(internal_desc, context->getEngine());
            if (is_output) {
                outputReorders.emplace_back(context->getReorderPrim(internal_desc, external_desc),
                                            PrimArgs{{DNNL_ARG_SRC, internal_mem}, {DNNL_ARG_DST, external_mem}});
            } else {
                inputReorders.emplace_back(context->getReorderPrim(external_desc, internal_desc),
                                           PrimArgs{{DNNL_ARG_SRC, external_mem}, {DNNL_ARG_DST, internal_mem}});
            }
            args[arg_id] = internal_mem;
        }
        return;
    }

    // p_canonical_desc is provided and not the same as exetrnal's desc, need canonicalization
    auto canonical_desc = *p_canonical_desc;
    dnnl::memory canonical_mem = dnnl::memory(canonical_desc, context->getEngine(), nullptr);
    canonicalizations.emplace_back(external_mem, canonical_mem);

    // canonical_mem matches primitive's needs, no further runtime reorder
    if (canonical_desc == internal_desc) {
        args[arg_id] = canonical_mem;
        return;
    }

    // further runtime reorder to/from canonicalized memory is still required
    dnnl::memory internal_mem(internal_desc, context->getEngine());
    if (is_output) {
        outputReorders.emplace_back(context->getReorderPrim(internal_desc, canonical_desc),
                                    PrimArgs{{DNNL_ARG_SRC, internal_mem}, {DNNL_ARG_DST, canonical_mem}});
    } else {
        inputReorders.emplace_back(context->getReorderPrim(canonical_desc, internal_desc),
                                   PrimArgs{{DNNL_ARG_SRC, canonical_mem}, {DNNL_ARG_DST, internal_mem}});
    }
    args[arg_id] = internal_mem;
}

void DnnlExecutor::setScratchPad() {
    if (pd.get_primitive_attr().get_scratchpad_mode() == dnnl::scratchpad_mode::user) {
        auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(pd.scratchpad_desc());
        scratchpadMem = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);
        args[DNNL_ARG_SCRATCHPAD] = scratchpadMem->GetPrimitive();
    } else {
        scratchpadMem = nullptr;
    }
}

dnnl::memory DnnlExecutor::addConstFolding(dnnl::memory external_mem,
                                           const dnnl::memory::desc* p_canonical_desc,
                                           std::string privateSrcKey,
                                           DnnlMemoryDescPtr expectedWeightDesc) {
    ConstFolding cf;
    cf.external_mem = external_mem;
    if (p_canonical_desc) {
        cf.canonical_desc = *p_canonical_desc;
    }
    cf.internal_mem = dnnl::memory(expectedWeightDesc->getDnnlDesc(), context->getEngine(), nullptr);
    cf.key = privateSrcKey + "_" + expectedWeightDesc->serializeFormat();
    constFoldings.push_back(cf);
    return cf.internal_mem;
}

void DnnlExecutor::doConstFolding(ConstFolding& cf) {
    // try find it in private weight cache
    auto it = privateWeightCache.find(cf.key);
    if (privateWeightCache.end() != it) {
        cf.internal_mem.set_data_handle(it->second->GetData());
        return;
    }

    // create or get from weight cache
    const auto& expectedDesc = cf.internal_mem.get_desc();
    auto canonical_src_desc = cf.external_mem.get_desc();
    // use canonical desc if user specified one
    if (cf.canonical_desc) {
        canonical_src_desc = cf.canonical_desc;
    }
    auto create = [&]() {
        auto cSrcDesc = DnnlExtensionUtils::makeDescriptor(canonical_src_desc);
        Memory srcMemory{context->getEngine()};
        srcMemory.Create(cSrcDesc, cf.external_mem.get_data_handle());
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine());
        _ptr->Create(DnnlExtensionUtils::makeDescriptor(expectedDesc));
        context->reorderData(srcMemory, *_ptr);
        DEBUG_LOG("ConstFolding ", cSrcDesc, " -> ", expectedDesc);
        return _ptr;
    };

    MemoryPtr ptr;
    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const std::string unique_name = cf.key + "_" + std::to_string(cf.external_mem.get_desc().get_size()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(cf.external_mem.get_data_handle()));
        ptr = *weightCache->findOrCreate(unique_name, create);
    } else {
        ptr = create();
    }
    cf.internal_mem.set_data_handle(ptr->GetData());
    privateWeightCache[cf.key] = ptr;
}

void DnnlExecutor::exec(dnnl::stream strm) {
    // update internal memory reinterpreted from external memory
    // using different descriptors
    for (auto& c : canonicalizations) {
        auto cur_handle = c.external_mem.get_data_handle();
        if (cur_handle != c.handle) {
            c.handle = cur_handle;
            c.canonical_mem.set_data_handle_no_pads_proc(cur_handle);
        }
    }

    // const folding is done once at first execution
    if (!constFolded) {
        for (auto& cf : constFoldings)
            doConstFolding(cf);
        constFolded = true;
    }

    // input reordering
    for (auto& s : inputReorders) {
        s.prim.execute(strm, s.args);
    }

    // main primitive
    prim.execute(strm, args);

    // output reordering
    for (auto& s : outputReorders) {
        s.prim.execute(strm, s.args);
    }
}

// dynamic batching:
//  given the actual batch existed in a fill-batch memory, some
//  primitive can save computations by limiting the calculations
//  only to the part actually exists.
//
//  this is done by changing the dim[0] (which is always batch in
//  oneDNN's canonical definition) of primitive's source/dest args
//  w/o touching strides.
//
void DnnlExecutor::setArgDynamicBatch(int arg_id, int newBatch) {
    auto param = args.find(arg_id);
    if (param != args.end()) {
        auto oldMem = param->second;
        dnnl::memory::desc newMemDesc(oldMem.get_desc());
        newMemDesc.data.dims[0] = newBatch;
        newMemDesc.data.padded_dims[0] = newBatch;
        dnnl::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());

        // if this primtive arg happened to be the target memory of a canonicalization
        // reorder, we also need to update that canonicalization entry.
        bool found = false;
        for (auto& c : canonicalizations) {
            if (c.canonical_mem == oldMem) {
                c.canonical_mem = newMem;
                found = true;
            }
        }
        // if it's not in `canonicalizations`, we need to add it into the list
        // because after dynamic batch is set, the newMem is actually doing
        // some canonicalization to the external memory and it's handle must
        // be updated according to external's handle at each execution
        if (!found) {
            canonicalizations.emplace_back(oldMem, newMem);
        }
        args.at(arg_id) = newMem;
    }
}

void DnnlExecutor::setDynamicBatch(int newBatch) {
    if (!prim) {
        IE_THROW() << "Can't set dynamic batch for node: " << name << ", because executor is not compiled";
    }

    if (needReordering()) {
        IE_THROW() << "Can't execute node " << name << " with dynamic batch via executor with reorders";
    }

    if (!args.empty()) {
        setArgDynamicBatch(DNNL_ARG_SRC, newBatch);
        setArgDynamicBatch(DNNL_ARG_DST, newBatch);
        setArgDynamicBatch(DNNL_ARG_DIFF_SRC, newBatch);
        setArgDynamicBatch(DNNL_ARG_DIFF_DST, newBatch);
    }
}

}  // namespace intel_cpu
}  // namespace ov
