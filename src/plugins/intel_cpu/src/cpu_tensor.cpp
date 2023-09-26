// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_tensor.h"
#include "ie_ngraph_utils.hpp"

#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

Tensor::Tensor(MemoryPtr memptr) : m_memptr{memptr} {
    OPENVINO_ASSERT(m_memptr != nullptr);

    // only support plain data format ncsp.
    auto memdesc = m_memptr->getDescPtr();
    OPENVINO_ASSERT(memdesc->hasLayoutType(LayoutType::ncsp), "intel_cpu::Tensor only supports memory with ncsp layout.");

    m_element_type = InferenceEngine::details::convertPrecision(memdesc->getPrecision());
}

void Tensor::set_shape(ov::Shape new_shape) {
    const auto& shape = m_memptr->getDescPtr()->getShape();
    if (shape.isStatic()) {
        DEBUG_LOG("tensor's memory object ", m_memptr.get(), ", ", vec2str(shape.getStaticDims()), " -> ", new_shape.to_string());
        if (shape.getStaticDims() == new_shape) return;
    }

    auto desc = m_memptr->getDescPtr();
    const auto newDesc = desc->cloneWithNewDims(new_shape, true);
    m_memptr->redefineDesc(newDesc);
}

const ov::element::Type& Tensor::get_element_type() const {
    return m_element_type;
}

const ov::Shape& Tensor::get_shape() const {
    auto& shape = m_memptr->getDescPtr()->getShape();
    OPENVINO_ASSERT(shape.isStatic(), "intel_cpu::Tensor has dynamic shape.");

    std::lock_guard<std::mutex> guard(m_lock);
    m_shape = ov::Shape{shape.getStaticDims()};
    return m_shape;
}

size_t Tensor::get_size() const {
    auto& desc = m_memptr->getDesc();
    return desc.getShape().getElementsCount();
}

size_t Tensor::get_byte_size() const {
    auto& desc = m_memptr->getDesc();
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
    auto& strides = blocked_desc->getStrides();
    m_strides.resize(strides.size());
    std::transform(strides.cbegin(), strides.cend(), m_strides.begin(), [this] (const size_t stride) {
        return stride * m_element_type.size();
    });
}

void* Tensor::data(const element::Type& element_type) const {
    if (element_type != element::undefined && element_type != element::dynamic) {
        OPENVINO_ASSERT(element_type == get_element_type(),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
    }
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
    return std::make_shared<Tensor>(mem);
}

}   // namespace intel_cpu
}   // namespace ov