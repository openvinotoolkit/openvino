// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "zero_memory.hpp"

namespace intel_npu {

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using custom allocator.
 * @details The implementation is simillar to the AllocatedTensor class from OV namespace.
 * @note Set_shape method throws an error in case re-allocation is needed but this is not supported by the driver.
 * There are two extra methods to notify the consumer if memory changed or not and to reset the flag.
 */
class ZeroTensor final : public ov::ITensor {
public:
    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const Config& config,
               const ov::element::Type element_type,
               const ov::Shape& shape,
               const bool isInput,
               const bool tensor_shared_with_user = false);

    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const std::shared_ptr<ov::ITensor>& user_tensor,
               const std::shared_ptr<ZeroTensor>& zero_tensor,
               const Config& config,
               const bool isInput,
               const bool dynamic_batch_value_changed = false);

    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const std::shared_ptr<ZeroRemoteTensor>& zero_remote_tensor,
               const Config& config,
               const bool isInput);

    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const std::shared_ptr<ov::ITensor>& user_tensor,
               const Config& config);

    void* data() override;
    void* data(const ov::element::Type& type) override;

    const void* data() const override;
    const void* data(const ov::element::Type& type) const override;

    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    void set_shape(ov::Shape new_shape) override;

    const ov::Strides& get_strides() const override;

    bool memory_address_changed();
    void reset_memory_flag();

    bool tensor_was_shared_with_user();
    void set_tensor_shared_with_user();
    bool update_command_list_arg();

    ~ZeroTensor();

private:
    static void initialize_elements(void* data, const ov::element::Type& element_type, const ov::Shape& shape);
    void update_strides() const;
    size_t get_capacity() const;
    size_t get_bytes_capacity() const;
    void destroy_elements(size_t begin_ind, size_t end_ind);
    void destroy_memory();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    Logger _logger;

    ov::element::Type _element_type;
    ov::Shape _shape;
    ov::Shape _capacity;
    mutable ov::Strides _strides;
    mutable std::once_flag _strides_once;
    std::unique_ptr<zeroMemory::HostMemAllocator> _allocator = nullptr;
    void* _ptr = nullptr;
    bool _reset_tensor_memory = false;
    bool _tensor_shared_with_user = false;
    bool _update_command_list_arg = false;

    std::shared_ptr<ov::ITensor> _imported_tensor;
};

}  // namespace intel_npu
