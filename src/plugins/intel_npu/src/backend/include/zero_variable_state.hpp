// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace intel_npu {

/**
 * @brief Interface for zero variable state implementation
 * @note In case the memory was allocated in the same level zero context use that memory, otherwise use memcpy at infer
 * time. Also, get correct data if remote tensor is used.
 */
class ZeroVariableState final : public ov::IVariableState {
public:
    explicit ZeroVariableState(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                               const std::string& name,
                               const ov::SoPtr<ov::ITensor>& tensor,
                               size_t tensor_index,
                               size_t related_tensor_index,
                               const Config& config,
                               bool external_memory_standard_allocation_supported);

    void set_state(const ov::SoPtr<ov::ITensor>& new_state) override;

    void reset() override;

    /**
     * @brief Get input tensor index used internally for the state
     */
    size_t get_tensor_index() const;

    /**
     * @brief Get output tensor index used internally for the state
     * @details The related tensors are defined by state input, state output pairs.
     */
    size_t get_related_tensor_index() const;

    /**
     * @brief Get acknowledgment if the tensor was updated
     */
    bool tensor_was_updated() const;

    /**
     * @brief Reset tensor updated flag
     */
    void reset_tensor_updated_flag();

    /**
     * @brief Get acknowledgment if the zero tensor was updated
     * @details In case the memory was allocated in the same level zero context update the zero tensor
     */
    bool zero_tensor_should_be_updated() const;

    /**
     * @brief Reset zero tensor updated flag
     */
    void reset_zero_tensor_updated_flag();

    /**
     * @brief Get acknowledgment if the zero tensor can be imported
     */
    bool zero_tensor_should_be_imported() const;

    /**
     * @brief Reset zero tensor imported flag
     */
    void reset_tensor_imported_flag();

    ~ZeroVariableState() override = default;

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    size_t _tensor_index;
    size_t _related_tensor_index;

    bool _tensor_updated = false;
    bool _zero_tensor_updated = false;
    bool _tensor_should_be_imported = false;

    bool _external_memory_standard_allocation_supported = false;

    const Config _config;
    Logger _logger;
};

}  // namespace intel_npu
