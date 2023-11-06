// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/runtime/make_tensor.hpp>

#include <nodes/common/cpu_convert.h>

#include "memory_state.h"

#include "dnnl_extension_utils.h"
#include "blob_factory.hpp"
#include "cpu_tensor.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

VariableStateDoubleBuffer::VariableStateDoubleBuffer(std::string name,
                                                     const MemBuilder& mem_build,
                                                     MemoryDescPtr external_desc,
                                                     MemoryCPtr init_val) :
    IVariableState{name}, m_external_desc{external_desc} {
    ResetPrimeMem(mem_build());
    ResetSecondMem(mem_build());
    m_internal_desc = PrimeMem()->getDescPtr();
    auto&& shape = m_internal_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        if (init_val) {
            PrimeMem()->load(*init_val);
        } else {
            PrimeMem()->nullify();
        }
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = ToStatic(m_internal_desc);
        PrimeMem()->redefineDesc(new_desc);
    }
}

void VariableStateDoubleBuffer::SetState(const Blob::Ptr& newState) {
    state = newState; // simply to extend the lifetime
    auto&& tensor_desc = state->getTensorDesc();
    if (PrimeMem()->getStaticDims() != tensor_desc.getDims()) {
        auto new_desc = m_internal_desc->cloneWithNewDims(tensor_desc.getDims());
        PrimeMem()->redefineDesc(new_desc);
    }
    auto blob_desc = MemoryDescUtils::convertToCpuBlockedMemoryDesc(tensor_desc);
    auto src = state->buffer().as<void*>();

    Memory mem(getEngine(), blob_desc, src);
    PrimeMem()->load(mem);
}

const dnnl::engine& VariableStateDoubleBuffer::getEngine() const {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

Blob::CPtr VariableStateDoubleBuffer::GetState() const {
    const auto& current_dims = PrimeMem()->getStaticDims();
    auto current_ext_desc = m_external_desc->cloneWithNewDims(current_dims);
    auto current_internal_desc = PrimeMem()->getDescPtr();

    if (current_ext_desc->isCompatible(*current_internal_desc)) {
        auto tensor = std::make_shared<Tensor>(PrimeMem());
        return tensor_to_blob({tensor, nullptr}); // TODO: shouldn't we provide the so ptr?
    }

    //test precision
    {
        auto internal_prc = current_internal_desc->getPrecision();
        auto tmp_desc = current_ext_desc->cloneWithNewPrecision(internal_prc);
        if (tmp_desc->isCompatible(*current_internal_desc)) {
            auto mem = std::make_shared<Memory>(getEngine(), current_ext_desc);
            size_t elements_to_convert = PrimeMem()->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            auto external_prc = current_ext_desc->getPrecision();

            cpu_convert(PrimeMem()->getData(), mem->getData(), internal_prc, external_prc, elements_to_convert);
            auto tensor = std::make_shared<Tensor>(mem);
            return tensor_to_blob({tensor, nullptr}); // TODO: shouldn't we provide the so ptr?
        }
    }

    //reorder
    auto mem = std::make_shared<Memory>(getEngine(), current_ext_desc);
    mem->load(*(PrimeMem()));
    auto tensor = std::make_shared<Tensor>(mem);
    return tensor_to_blob({tensor, nullptr}); // TODO: shouldn't we provide the so ptr?
}

void VariableStateDoubleBuffer::Reset() {
    auto new_desc = ToStatic(m_internal_desc);
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

MemoryDescPtr VariableStateDoubleBuffer::InternalDesc() const {
    return m_internal_desc;
}

}  // namespace intel_cpu
}  // namespace ov
