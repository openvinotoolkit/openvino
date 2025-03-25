// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "openvino/runtime/itensor.hpp"

namespace ov {
namespace intel_cpu {

class Tensor : public ITensor {
public:
    // Only plain data format is supported.
    explicit Tensor(MemoryPtr memptr);

    void set_shape(ov::Shape shape) override;

    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    size_t get_size() const override;

    size_t get_byte_size() const override;

    const ov::Strides& get_strides() const override;

    void* data(const element::Type& type) const override;

    MemoryPtr get_memory() {
        return m_memptr;
    }

private:
    void update_strides() const;

    MemoryPtr m_memptr;

    ov::element::Type m_element_type;
    mutable ov::Shape m_shape;
    mutable ov::Strides m_strides;
    mutable std::mutex m_lock;
};

std::shared_ptr<ITensor> make_tensor(MemoryPtr mem);

}  // namespace intel_cpu
}  // namespace ov
