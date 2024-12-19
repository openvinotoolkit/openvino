// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using custom allocator.
 * @details The implementation is simillar with the AllocatedTensor class from OV repository.
 * @note Set_shape method throw an error in case re-allocation is needed but this is not supported by the driver.
 * There are two extra methods to notify the consumer if memorey changed or not and to reset the flag.
 */
class ZeroTensor final : public ov::ITensor {
public:
    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const ov::element::Type element_type,
               const ov::Shape& shape,
               const ov::Allocator& allocator);

    void* data(const ov::element::Type& type = {}) const override;

    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    void set_shape(ov::Shape new_shape) override;

    const ov::Strides& get_strides() const override;

    bool memory_address_changed();
    void reset_memory_flag();

    ~ZeroTensor();

private:
    static void initialize_elements(void* data, const ov::element::Type& element_type, const ov::Shape& shape);
    void update_strides() const;
    size_t get_capacity() const;
    size_t get_bytes_capacity() const;
    void destroy_elements(size_t begin_ind, size_t end_ind);
    void destroy_memory();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    ov::element::Type _element_type;
    ov::Shape _shape;
    ov::Shape _capacity;
    mutable ov::Strides _strides;
    mutable std::once_flag _strides_once;
    ov::Allocator _allocator;
    void* _ptr = nullptr;
    bool _reset_tensor_memory = false;
};

}  // namespace intel_npu
