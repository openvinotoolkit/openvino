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

VariableStateDoubleBuffer::VariableStateDoubleBuffer(std::string name,
                                                     const MemBuilder& mem_build,
                                                     MemoryDescPtr external_desc,
                                                     MemoryCPtr init_val) :
    IVariableState{name}, m_external_desc{external_desc} {
    reset_prime_mem(mem_build());
    reset_second_mem(mem_build());
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
        auto new_desc = ToStatic(m_internal_desc);
        prime_mem()->redefineDesc(new_desc);
    }
}

void VariableStateDoubleBuffer::set_state(const ov::SoPtr<ov::ITensor>& state) {
    m_state = state; // simply to extend the lifetime
    auto state_desc = MemoryDescUtils::generateCpuBlockedMemoryDesc(m_state);

    const auto& shape = state_desc->getShape();

    if (prime_mem()->getShape() != shape) {
        auto new_desc = m_internal_desc->cloneWithNewDims(shape.getStaticDims());
        prime_mem()->redefineDesc(new_desc);
    }

    auto src = m_state->data();

    Memory mem(get_engine(), state_desc, src);
    prime_mem()->load(mem);
}

const dnnl::engine& VariableStateDoubleBuffer::get_engine() const {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

const ov::SoPtr<ov::ITensor>& VariableStateDoubleBuffer::get_state() const {
    //TODO , in general case must be synchronized
    const auto& current_dims = prime_mem()->getStaticDims();
    auto current_ext_desc = m_external_desc->cloneWithNewDims(current_dims);
    auto current_internal_desc = prime_mem()->getDescPtr();

    if (current_ext_desc->isCompatible(*current_internal_desc)) {
        m_converted_state = std::make_shared<Tensor>(prime_mem());
        return m_converted_state; // TODO: shouldn't we provide the so ptr?
    }

    //test precision
    {
        auto internal_prc = current_internal_desc->getPrecision();
        auto tmp_desc = current_ext_desc->cloneWithNewPrecision(internal_prc);
        if (tmp_desc->isCompatible(*current_internal_desc)) {
            auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
            size_t elements_to_convert = prime_mem()->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
            auto external_prc = current_ext_desc->getPrecision();

            cpu_convert(prime_mem()->getData(), mem->getData(), internal_prc, external_prc, elements_to_convert);
            m_converted_state = std::make_shared<Tensor>(mem);
            return m_converted_state;
        }
    }

    //reorder
    auto mem = std::make_shared<Memory>(get_engine(), current_ext_desc);
    mem->load(*(prime_mem()));
    m_converted_state = std::make_shared<Tensor>(mem);
    return m_converted_state;
}

void VariableStateDoubleBuffer::reset() {
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

}  // namespace intel_cpu
}  // namespace ov
