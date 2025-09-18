// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "zero_tensor.hpp"

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
                               const std::shared_ptr<ZeroTensor>& zero_tensor,
                               size_t tensor_index,
                               size_t related_tensor_index,
                               const Config& config);

    void set_state(const ov::SoPtr<ov::ITensor>& new_state) override;

    void reset() override;

    ov::SoPtr<ov::ITensor> get_state() const override;

    /**
     * @brief Get user state to not change the state of the tensor through get_state()
     */
    ov::SoPtr<ov::ITensor> get_user_state() const;

    std::shared_ptr<ZeroTensor> get_zero_state() const;

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
     * @brief Get acknowledgment if state was updated
     */
    bool state_update_pending() const;

    /**
     * @brief Reset state updated flag
     */
    void clear_state_update_pending();

    /**
     * @brief Get acknowledgment if the zero state was updated
     * @details In case the memory was allocated in the same level zero context update the zero state
     */
    bool zero_state_update_pending() const;

    /**
     * @brief Reset zero state updated flag
     */
    void clear_zero_state_update_pending();

    ~ZeroVariableState() override = default;

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    size_t _tensor_index;
    size_t _related_tensor_index;

    std::shared_ptr<ZeroTensor> _zero_state;

    bool _is_state_updated = false;
    bool _is_zero_state_update_needed = false;

    const Config _config;
    Logger _logger;
};

}  // namespace intel_npu
