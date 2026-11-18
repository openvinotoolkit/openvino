// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

// Keeps the source tensor alive and exposes it for backend allocation recognition.
class StableTensorView final : public ov::ITensor {
public:
    StableTensorView(ov::SoPtr<ov::ITensor> owner, const ov::Shape& shape) : m_owner(std::move(owner)) {
        OPENVINO_ASSERT(m_owner);
        OPENVINO_ASSERT(ov::util::get_memory_size_safe(m_owner->get_element_type(), shape));
        OPENVINO_ASSERT(m_owner->get_size() == ov::shape_size(shape));
        m_view = ov::make_tensor(m_owner->get_element_type(), shape, m_owner->data());
        OPENVINO_ASSERT(m_view->get_byte_size() == m_owner->get_byte_size());
    }

    const ov::SoPtr<ov::ITensor>& owner() const {
        return m_owner;
    }

    void set_shape(ov::Shape shape) override {
        m_view->set_shape(std::move(shape));
    }
    const ov::element::Type& get_element_type() const override {
        return m_view->get_element_type();
    }
    const ov::Shape& get_shape() const override {
        return m_view->get_shape();
    }
    const ov::Strides& get_strides() const override {
        return m_view->get_strides();
    }
    void* data() override {
        return m_view->data();
    }
    const void* data() const override {
        return m_view->data();
    }
    void* data(const ov::element::Type& type) override {
        return m_view->data(type);
    }
    const void* data(const ov::element::Type& type) const override {
        return m_view->data(type);
    }
    void* data_rw() override {
        return m_view->data_rw();
    }
    void* data_rw(const ov::element::Type& type) override {
        return m_view->data_rw(type);
    }

private:
    ov::SoPtr<ov::ITensor> m_owner;
    std::shared_ptr<ov::ITensor> m_view;
};

}  // namespace intel_npu
