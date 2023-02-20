// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "tensor_data_adapter.hpp"

namespace cldnn {
/**
 * @brief cldnn Memory adapter to access its data as Tensor.
 *
 * This adapter takes data ownership by store the memory lock and release when object will be destroyed.
 *
 * @note This is example how to use it with tensor data accessor function instead creating a map.
 * Could be part of intel_gpu/runtime/include/memory.hpp but core dev API headers are not visible there.
 */
class MemoryAdapter : public ov::ITensorDataAdapter {
public:
    MemoryAdapter(cldnn::memory::ptr ptr, const cldnn::stream& stream);

    ov::element::Type_t get_element_type() const override;
    size_t get_size() const override;
    const void* data() const override;

private:
    cldnn::memory::ptr m_ptr;                                  //!< Pointer to cldnn::memory.
    cldnn::mem_lock<uint8_t, mem_lock_type::read> m_ptr_lock;  //!< Store cldnn memory lock.
};
}  // namespace cldnn
