// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/host_tensor.hpp"

#include <cstring>
#include <memory>

#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;
OPENVINO_SUPPRESS_DEPRECATED_START

static const size_t alignment = 64;

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape)),
      m_memory_pointer(memory_pointer) {
    if (get_partial_shape().is_static() && get_element_type().is_static()) {
        allocate_buffer();
    } else {
        m_buffer_size = 0;
    }
}

runtime::HostTensor::HostTensor(const element::Type& element_type, const Shape& shape)
    : HostTensor(element_type, shape, nullptr) {}

runtime::HostTensor::HostTensor(const element::Type& element_type, const PartialShape& partial_shape)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, partial_shape)),
      m_buffer_size(0) {
    // Defer allocation until ptr is requested
}

runtime::HostTensor::HostTensor() : HostTensor(element::dynamic, PartialShape::dynamic()) {}

NGRAPH_SUPPRESS_DEPRECATED_START
runtime::HostTensor::HostTensor(const Output<Node>& value)
    : HostTensor(value.get_element_type(), value.get_partial_shape()) {}
NGRAPH_SUPPRESS_DEPRECATED_END

void runtime::HostTensor::allocate_buffer() {
    NGRAPH_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(get_partial_shape().is_static(),
                 "Attempt to allocate buffer for tensor with partial shape: ",
                 get_partial_shape());
    NGRAPH_CHECK(get_element_type().is_static(),
                 "Attempt to allocate buffer for tensor with dynamic type: ",
                 get_element_type());
    m_buffer_size = m_descriptor->size();
    if (m_memory_pointer != nullptr) {
        m_aligned_buffer_pool = m_memory_pointer;
    } else {
        // Add 1 so that even for zero-sized tensor we get at least 1 byte
        size_t allocation_size = m_buffer_size + alignment + 1;
        uint8_t* allocated_buffer_pool = static_cast<uint8_t*>(ngraph_malloc(allocation_size));
        m_allocated_buffer_pool = allocated_buffer_pool;
        size_t mod = size_t(allocated_buffer_pool) % alignment;
        if (mod == 0) {
            m_aligned_buffer_pool = allocated_buffer_pool;
        } else {
            m_aligned_buffer_pool = (allocated_buffer_pool + alignment - mod);
        }
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

NGRAPH_SUPPRESS_DEPRECATED_START
runtime::HostTensor::HostTensor(const std::shared_ptr<op::v0::Constant>& constant) : HostTensor() {
    initialize(constant);
}
NGRAPH_SUPPRESS_DEPRECATED_END

void runtime::HostTensor::initialize(const std::shared_ptr<op::v0::Constant>& constant) {
    set_element_type(constant->get_output_element_type(0));
    set_shape(constant->get_output_shape(0));
    memcpy(get_data_ptr(), constant->get_data_ptr(), get_size_in_bytes());
}

runtime::HostTensor::~HostTensor() {
    NGRAPH_SUPPRESS_DEPRECATED_START
    if (m_allocated_buffer_pool != nullptr) {
        ngraph_free(m_allocated_buffer_pool);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

void* runtime::HostTensor::get_data_ptr() {
    if (!m_aligned_buffer_pool) {
        allocate_buffer();
    }
    return m_aligned_buffer_pool;
}

const void* runtime::HostTensor::get_data_ptr() const {
    NGRAPH_CHECK(m_aligned_buffer_pool, "Buffer not initialized");
    return m_aligned_buffer_pool;
}

void runtime::HostTensor::write(const void* source, size_t n) {
    void* target = get_data_ptr();
    if (n != m_buffer_size) {
        throw out_of_range("partial tensor write not supported");
    }
    if (n > 0) {
        if (!source) {
            throw runtime_error("nullptr passed to HostTensor::write");
        }
        memcpy(target, source, n);
    }
}

void runtime::HostTensor::read(void* target, size_t n) const {
    const void* source = get_data_ptr();
    if (n != m_buffer_size) {
        throw out_of_range("partial tensor read access not supported");
    }
    if (n > 0) {
        if (!target) {
            throw runtime_error("nullptr passed to HostTensor::read");
        }
        memcpy(target, source, n);
    }
}

bool runtime::HostTensor::get_is_allocated() const {
    return m_aligned_buffer_pool != nullptr;
}

void runtime::HostTensor::set_element_type(const element::Type& element_type) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    NGRAPH_CHECK(get_element_type().is_dynamic() || get_element_type() == element_type,
                 "Can not change a static element type");
    m_descriptor->set_element_type(element_type);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void runtime::HostTensor::set_shape(const Shape& shape) {
    NGRAPH_CHECK(PartialShape(shape).refines(get_partial_shape()) ||
                     (m_descriptor->get_partial_shape().is_static() &&
                      m_descriptor->get_partial_shape().to_shape() == ov::Shape{0}),
                 "Allocation shape ",
                 shape,
                 " must be compatible with the partial shape: ",
                 get_partial_shape());
    m_descriptor->m_partial_shape = shape;
    m_descriptor->m_shape_changed = true;
}

void runtime::HostTensor::set_unary(const HostTensorPtr& arg) {
    set_element_type(arg->get_element_type());
    set_shape(arg->get_partial_shape().get_shape());
}

void runtime::HostTensor::set_broadcast(const op::AutoBroadcastSpec& autob,
                                        const HostTensorPtr& arg0,
                                        const HostTensorPtr& arg1) {
    element::Type element_type = arg0->get_element_type();
    NGRAPH_CHECK(element::Type::merge(element_type, element_type, arg1->get_element_type()),
                 "Argument element types are inconsistent.");
    set_broadcast(autob, arg0, arg1, element_type);
}

void runtime::HostTensor::set_broadcast(const op::AutoBroadcastSpec& autob,
                                        const HostTensorPtr& arg0,
                                        const HostTensorPtr& arg1,
                                        const element::Type& element_type) {
    set_element_type(element_type);

    PartialShape pshape = arg0->get_partial_shape();
    if (autob.m_type == op::AutoBroadcastType::NONE) {
        NGRAPH_CHECK(PartialShape::merge_into(pshape, arg1->get_partial_shape()), "Argument shapes are inconsistent.");
    } else if (autob.m_type == op::AutoBroadcastType::NUMPY || autob.m_type == op::AutoBroadcastType::PDPD) {
        NGRAPH_CHECK(PartialShape::broadcast_merge_into(pshape, arg1->get_partial_shape(), autob),
                     "Argument shapes are inconsistent.");
    } else {
        NGRAPH_CHECK(false, "Unsupported auto broadcast specification");
    }
    set_shape(pshape.get_shape());
}
