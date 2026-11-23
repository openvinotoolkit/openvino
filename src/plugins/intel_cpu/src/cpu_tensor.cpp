// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_tensor.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#include "cpu_memory.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/util/math_util.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {
constexpr bool is_pointer_representable(const ov::element::Type& tensor_type, const ov::element::Type& type) {
    return any_of(type, ov::element::dynamic, tensor_type);
}
}  // namespace

Tensor::Tensor(MemoryPtr memptr) : m_memptr{std::move(memptr)} {
    OPENVINO_ASSERT(m_memptr);

    // only support plain data format ncsp.
    auto memdesc = m_memptr->getDescPtr();
    OPENVINO_ASSERT(memdesc->hasLayoutType(LayoutType::ncsp),
                    "intel_cpu::Tensor only supports memory with ncsp layout.");

    m_element_type = memdesc->getPrecision();
}

void Tensor::set_shape(ov::Shape new_shape) {
    const auto& shape = m_memptr->getDescPtr()->getShape();
    if (shape.isStatic()) {
        DEBUG_LOG("tensor's memory object ",
                  m_memptr.get(),
                  ", ",
                  vec2str(shape.getStaticDims()),
                  " -> ",
                  new_shape.to_string());
        if (shape.getStaticDims() == new_shape) {
            return;
        }
    }

    auto desc = m_memptr->getDescPtr();
    const auto newDesc = desc->cloneWithNewDims(new_shape, true);
    m_memptr->redefineDesc(newDesc);
}

const ov::element::Type& Tensor::get_element_type() const {
    return m_element_type;
}

const ov::Shape& Tensor::get_shape() const {
    const auto& shape = m_memptr->getDescPtr()->getShape();
    OPENVINO_ASSERT(shape.isStatic(), "intel_cpu::Tensor has dynamic shape.");

    std::lock_guard<std::mutex> guard(m_lock);
    m_shape = ov::Shape{shape.getStaticDims()};
    return m_shape;
}

size_t Tensor::get_size() const {
    const auto& desc = m_memptr->getDesc();
    return desc.getShape().getElementsCount();
}

size_t Tensor::get_byte_size() const {
    const auto& desc = m_memptr->getDesc();
    return desc.getCurrentMemSize();
}

const ov::Strides& Tensor::get_strides() const {
    OPENVINO_ASSERT(m_memptr->getDescPtr()->isDefined(), "intel_cpu::Tensor requires memory with defined strides.");

    std::lock_guard<std::mutex> guard(m_lock);
    update_strides();
    return m_strides;
}

void Tensor::update_strides() const {
    auto blocked_desc = m_memptr->getDescWithType<BlockedMemoryDesc>();
    OPENVINO_ASSERT(blocked_desc, "not a valid blocked memory descriptor.");
    const auto& strides = blocked_desc->getStrides();
    const auto bitwidth = m_element_type.bitwidth();
    m_strides.resize(strides.size());
    std::ranges::transform(strides, m_strides.begin(), [bitwidth](const size_t stride) {
        size_t bits = 0;
        const bool is_overflow = ov::util::mul_overflow(stride, bitwidth, bits);
        OPENVINO_ASSERT(!is_overflow, "Tensor::update_strides: stride to bits conversion overflow");
        return bits / CHAR_BIT;
    });
    if (!m_strides.empty() && bitwidth < CHAR_BIT && strides.back() != 0) {
        // the innermost dimension's byte stride is a non-addressable placeholder for sub-byte
        // types (unless the tensor is degenerate, i.e. its element stride is already 0)
        m_strides.back() = m_element_type.size();
    }
}

void* Tensor::data() {
    return m_memptr->getData();
}

void* Tensor::data(const element::Type& element_type) {
    OPENVINO_ASSERT(is_pointer_representable(get_element_type(), element_type),
                    "Tensor data with element type ",
                    get_element_type(),
                    ", is not representable as pointer to ",
                    element_type);
    return m_memptr->getData();
}

const void* Tensor::data() const {
    return m_memptr->getData();
}

const void* Tensor::data(const element::Type& element_type) const {
    OPENVINO_ASSERT(is_pointer_representable(get_element_type(), element_type),
                    "Tensor data with element type ",
                    get_element_type(),
                    ", is not representable as pointer to ",
                    element_type);
    return m_memptr->getData();
}

void* Tensor::data_rw() {
    return m_memptr->getData();
}

void* Tensor::data_rw(const element::Type& element_type) {
    OPENVINO_ASSERT(is_pointer_representable(get_element_type(), element_type),
                    "Tensor data with element type ",
                    get_element_type(),
                    ", is not representable as pointer to ",
                    element_type);
    return m_memptr->getData();
}

/**
 * @brief Creates tensor on graph memory
 *
 * @param mem Memory object
 *
 * @return Shared pointer to tensor interface
 */
std::shared_ptr<ITensor> make_tensor(MemoryPtr mem) {
    return std::make_shared<Tensor>(std::move(mem));
}

}  // namespace ov::intel_cpu
