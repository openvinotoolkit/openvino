// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <nodes/common/cpu_convert.h>

#include "memory_state.h"

#include "dnnl_extension_utils.h"
#include "blob_factory.hpp"
#include "cpu_tensor.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

VariableStateBase::VariableStateBase(const std::string& name, const MemoryDescPtr& external_desc) :
    IVariableState{name} , m_external_desc{external_desc} {}

MemoryDescPtr VariableStateBase::to_static(const MemoryDescPtr& desc) {
    if (!desc->isDefined()) {
        auto&& current_dims = desc->getShape().getDims();
        VectorDims new_dims(current_dims.size());
        std::transform(current_dims.begin(), current_dims.end(), new_dims.begin(), [](Dim x) {
            return x == Shape::UNDEFINED_DIM ? 0 : x; });

        return desc->cloneWithNewDims(new_dims, true);
    }
    return desc;
}

const dnnl::engine& VariableStateBase::get_engine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

void VariableStateBase::set_state(const ov::SoPtr<ov::ITensor>& state) {
    m_state = state; // simply to extend the lifetime
    auto state_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(m_state);

    const auto& shape = state_desc->getShape();

    if (input_mem()->getShape() != shape) {
        auto new_desc = internal_desc()->cloneWithNewDims(shape.getStaticDims());
        input_mem()->redefineDesc(new_desc);
    }

    auto src = m_state->data();

    Memory mem(get_engine(), state_desc, src);
    input_mem()->load(mem);
}

ov::SoPtr<ov::ITensor> VariableStateBase::get_state() const {
    const auto& current_dims = internal_state_mem()->getStaticDims();
    auto current_ext_desc = m_external_desc->cloneWithNewDims(current_dims);
    auto current_internal_desc = internal_state_mem()->getDescPtr();

    if (current_ext_desc->isCompatible(*current_internal_desc)) {
        return std::make_shared<Tensor>(internal_state_mem());
    }

    //test precision
    {
        auto internal_prc = current_internal_desc->getPrecision();
        auto tmp_desc = current_ext_desc->cloneWithNewPrecision(internal_prc);
        if (tmp_desc->isCompatible(*current_internal_desc)) {
            auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
            size_t elements_to_convert = internal_state_mem()->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            auto external_prc = current_ext_desc->getPrecision();

            cpu_convert(internal_state_mem()->getData(), mem->getData(), internal_prc, external_prc, elements_to_convert);
            return std::make_shared<Tensor>(mem);
        }
    }

    //reorder
    auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
    mem->load(*(internal_state_mem()));
    return std::make_shared<Tensor>(mem);
}

VariableStateDoubleBuffer::VariableStateDoubleBuffer(const std::string& name,
                                                     const MemoryPtr& first_buffer,
                                                     const MemoryPtr& second_buffer,
                                                     const MemoryDescPtr& external_desc,
                                                     const MemoryCPtr& init_val) :
    VariableStateBase(name, external_desc) {
    OPENVINO_ASSERT(first_buffer && second_buffer);
    reset_prime_mem(first_buffer);
    reset_second_mem(second_buffer);
    m_internal_desc = prime_mem()->getDescPtr();
    auto&& shape = m_internal_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        if (init_val) {
            prime_mem()->load(*init_val);
        } else {
            prime_mem()->nullify();
        }
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = to_static(m_internal_desc);
        prime_mem()->redefineDesc(new_desc);
    }
}

void VariableStateDoubleBuffer::reset() {
    auto new_desc = to_static(m_internal_desc);
    for (auto&& mem : m_internal_mem) {
        if (mem) {
            mem->redefineDesc(new_desc);
            mem->nullify();
        }
    }
}

void VariableStateDoubleBuffer::commit() {
    buffer_num ^= 0x01;
}

MemoryPtr VariableStateDoubleBuffer::input_mem() {
    return prime_mem();
}

MemoryPtr VariableStateDoubleBuffer::output_mem() {
    return second_mem();
}

MemoryDescPtr VariableStateDoubleBuffer::internal_desc() const {
    return m_internal_desc;
}

MemoryPtr VariableStateDoubleBuffer::internal_state_mem() const {
    return prime_mem();
}

VariableStateSingleBuffer::VariableStateSingleBuffer(const std::string& name,
                                                     const MemoryPtr& buffer,
                                                     const MemoryDescPtr& external_desc,
                                                     const MemoryCPtr& init_val) :
    VariableStateBase(name, external_desc) {
    OPENVINO_ASSERT(buffer);
    m_internal_mem = buffer;
    m_internal_desc = m_internal_mem->getDescPtr();
    auto&& shape = m_internal_desc->getShape();
    //TODO what if by some reason we already have internal static state while the node is dynamic, is it even possible?

    if (shape.isStatic()) {
        if (init_val) {
            m_internal_mem->load(*init_val);
        } else {
            m_internal_mem->nullify();
        }
    } else {
        //in the case of the original desc has dynamic shape we create an empty tensor
        auto new_desc = to_static(m_internal_desc);
        m_internal_mem->redefineDesc(new_desc);
    }
}

void VariableStateSingleBuffer::reset() {
    auto new_desc = to_static(m_internal_desc);
    m_internal_mem->redefineDesc(new_desc);
    m_internal_mem->nullify();
}

MemoryPtr VariableStateSingleBuffer::input_mem() {
    return m_internal_mem;
}

MemoryPtr VariableStateSingleBuffer::output_mem() {
    return m_internal_mem;
}

MemoryDescPtr VariableStateSingleBuffer::internal_desc() const {
    return m_internal_desc;
}

MemoryPtr VariableStateSingleBuffer::internal_state_mem() const {
    return m_internal_mem;
}

void VariableStateSingleBuffer::commit() {
    //nothing to do
}

}  // namespace intel_cpu
}  // namespace ov
