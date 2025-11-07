// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

/**
 * @brief ZeroTensor API holding NPU device memory
 * It keeps a data pointer allocated in the same Level Zero context.
 */
class ZeroTensor final : public ov::ITensor {
public:
    /**
     * @brief Constructs a ZeroTensor with the specified element type and shape. Allocates internal storage in the given
     * level zero context.
     * @param init_structs Shared pointer to the ZeroInitStructHolder instance that will provide the level zero context.
     * @param config NPU plugin configuration
     * @param type Data type of tensor elements
     * @param shape Tensor shape
     * @param is_input Indicates if the tensor is used as a network input ( true) or output (false)
     */
    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const Config& config,
               const ov::element::Type element_type,
               const ov::Shape& shape,
               const bool is_input);

    /**
     * @brief Creates a ZeroTensor from the given tensor. This constructor will throw if the memory of the given tensor
     * is not allocated in the level zero context specified through init_structs or in case the memory cannot be
     * imported in that context ( to be implemented). ZeroTensor will keep a reference to the source tensor.
     * @param init_structs Shared pointer to ZeroInitStructsHolder
     * @param config NPU plugin configuration
     * @param user_tensor Tensor to create ZeroTensor from
     */
    ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
               const Config& config,
               const ov::SoPtr<ov::ITensor>& user_tensor);

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

    void prevent_reuse();
    bool can_be_reused();

    ~ZeroTensor() override = default;

private:
    void update_strides() const;
    size_t get_capacity() const;
    size_t get_bytes_capacity() const;

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    Logger _logger;

    ov::element::Type _element_type;
    ov::Shape _shape;
    ov::Shape _capacity;
    mutable ov::Strides _strides;
    mutable std::once_flag _strides_once;
    void* _ptr = nullptr;
    bool _reset_tensor_memory = false;
    bool _is_input = false;
    bool _can_be_reused = false;

    ov::SoPtr<ov::ITensor> _user_tensor;

    std::shared_ptr<ZeroMem> _mem_ref;
};

}  // namespace intel_npu
