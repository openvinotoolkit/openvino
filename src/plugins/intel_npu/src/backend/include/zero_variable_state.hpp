// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/variable_state.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/network_metadata.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

class ZeroVariableState final : public VariableState {
public:
    explicit ZeroVariableState(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                               const IODescriptor& descriptor,
                               const ov::SoPtr<ov::ITensor>& tensor,
                               size_t index,
                               const Config& config);

    void set_state(const ov::SoPtr<ov::ITensor>& new_state) override;

    void reset() override;

    size_t get_index() const;
    const IODescriptor& get_descriptor() const;

    bool tensor_was_updated() const;
    void reset_tensor_updated_flag();

    bool zero_tensor_should_be_updated() const;
    void reset_zero_tensor_updated_flag();

    ~ZeroVariableState() override = default;

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    IODescriptor _descriptor;
    size_t _index;

    bool _tensor_updated = false;
    bool _zero_tensor_updated = false;

    Logger _logger;
};

}  // namespace intel_npu
