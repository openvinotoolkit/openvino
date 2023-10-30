// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/make_tensor.hpp>

#include "memory_state.h"

#include "dnnl_extension_utils.h"
#include "blob_factory.hpp"
#include "cpu_tensor.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

VariableStateDoubleBuffer::VariableStateDoubleBuffer(std::string name,
                                                     const MemBuilder& mem_build,
                                                     MemoryCPtr init_val) :
    IVariableState{name} {
    ResetPrimeMem(mem_build());
    ResetSecondMem(mem_build());
    m_desc = PrimeMem()->getDescPtr();
    auto&& shape = m_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        if (init_val) {
            PrimeMem()->load(*init_val);
        } else {
            PrimeMem()->nullify();
        }
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = ToStatic(m_desc);
        PrimeMem()->redefineDesc(new_desc);
    }
}

void VariableStateDoubleBuffer::SetState(const Blob::Ptr& newState) {
    state = newState; // simply to extend the lifetime
    auto&& tensor_desc = state->getTensorDesc();
    if (PrimeMem()->getStaticDims() != tensor_desc.getDims()) {
        auto new_desc = m_desc->cloneWithNewDims(tensor_desc.getDims());
        PrimeMem()->redefineDesc(new_desc);
    }
    auto blob_desc = MemoryDescUtils::convertToCpuBlockedMemoryDesc(tensor_desc);
    auto src = state->buffer().as<void*>();

    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    Memory mem(eng, blob_desc, src);
    PrimeMem()->load(mem);
}

Blob::CPtr VariableStateDoubleBuffer::GetState() const {
    auto tensor = std::make_shared<Tensor>(PrimeMem());
    return tensor_to_blob({tensor, nullptr}); // TODO: shouldn't we provide the so ptr?
}

void VariableStateDoubleBuffer::Reset() {
    auto new_desc = ToStatic(m_desc);
    for (auto&& mem : m_internal_mem) {
        if (mem) {
            mem->redefineDesc(new_desc);
            mem->nullify();
        }
    }
}

MemoryDescPtr VariableStateDoubleBuffer::ToStatic(const MemoryDescPtr& desc) {
    if (!desc->isDefined()) {
        auto&& current_dims = desc->getShape().getDims();
        VectorDims new_dims(current_dims.size());
        std::transform(current_dims.begin(), current_dims.end(), new_dims.begin(), [](Dim x) {
            return x == Shape::UNDEFINED_DIM ? 0 : x; });

        return desc->cloneWithNewDims(new_dims, true);
    }
    return desc;
}

void VariableStateDoubleBuffer::Commit() {
    buffer_num ^= 0x01;
}

MemoryPtr VariableStateDoubleBuffer::InputMem() {
    return PrimeMem();
}

MemoryPtr VariableStateDoubleBuffer::OutputMem() {
    return SecondMem();
}

MemoryDescPtr VariableStateDoubleBuffer::OriginalDesc() const {
    return m_desc;
}

}  // namespace intel_cpu
}  // namespace ov
