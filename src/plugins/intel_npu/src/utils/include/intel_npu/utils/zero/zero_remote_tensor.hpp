// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_memory.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"

namespace intel_npu {

class ZeroRemoteTensor final : public ov::IRemoteTensor {
public:
    ZeroRemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                     const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                     const ov::element::Type& element_type,
                     const ov::Shape& shape,
                     ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED,
                     ov::intel_npu::MemType mem_type = ov::intel_npu::MemType::L0_INTERNAL_BUF,
                     const void* mem = nullptr,
                     const std::optional<ov::intel_npu::FileDescriptor>& file_descriptor = std::nullopt);

    /**
     * @brief Returns additional information associated with tensor
     * @return Map of property names to properties
     */
    const ov::AnyMap& get_properties() const override;

    /**
     * @brief Returns device name
     * @return Device name
     */
    const std::string& get_device_name() const override;

    /**
     * @brief Set new shape for tensor
     * @note Allocation of a bigger tensor is not possible
     * @param shape A new shape
     */
    void set_shape(ov::Shape shape) override;

    /**
     * @return A tensor element type
     */
    const ov::element::Type& get_element_type() const override;

    /**
     * @return A tensor shape
     */
    const ov::Shape& get_shape() const override;

    /**
     * @return Tensor's strides in bytes
     */
    const ov::Strides& get_strides() const override;

    /**
     * @return The remote context
     */
    std::shared_ptr<ov::IRemoteContext> get_context() const;

    void* get_original_memory() const;
    ze_context_handle_t get_zero_context_handle() const;

    ~ZeroRemoteTensor() override = default;

private:
    void allocate(const size_t bytes);
    bool deallocate() noexcept;
    bool is_allocated() const noexcept;
    void update_strides();
    void update_properties();
    void copy_file_data_to_level_zero_memory(const size_t size_to_read);

    std::shared_ptr<ov::IRemoteContext> _context;
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    ov::element::Type _element_type;
    ov::Shape _shape;
    ov::Shape _capacity;
    ov::Strides _strides{};
    ov::AnyMap _properties;

    Logger _logger;

    ov::intel_npu::TensorType _tensor_type;
    ov::intel_npu::MemType _mem_type;
    std::optional<ov::intel_npu::FileDescriptor> _file_descriptor;
    const void* _mem = nullptr;
    void* _data = nullptr;

    ov::Tensor _mmap_tensor;
    std::shared_ptr<ZeroMem> _host_memory;
};

inline bool is_remote_tensor(const std::shared_ptr<ov::ITensor>& tensor) {
    return std::dynamic_pointer_cast<ZeroRemoteTensor>(tensor) != nullptr;
}

}  // namespace intel_npu
