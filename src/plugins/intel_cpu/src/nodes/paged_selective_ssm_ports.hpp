// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ov::intel_cpu {

enum class PagedSelectiveSSMInputPort : uint8_t {
    A,
    TimeStep,
    InputProjection,
    Input,
    OutputProjection,
    State,
    SubsequenceBegins,
    BlockIndices,
    BlockIndicesBegins,
    NumProcessedTokens,
    CacheInterval,
    Count,
};

constexpr size_t input_port_index(PagedSelectiveSSMInputPort port) noexcept {
    return static_cast<size_t>(port);
}

inline constexpr size_t paged_ssm_input_count = input_port_index(PagedSelectiveSSMInputPort::Count);

enum class PagedSelectiveSSMOutputPort : uint8_t {
    Output,
    Count,
};

constexpr size_t output_port_index(PagedSelectiveSSMOutputPort port) noexcept {
    return static_cast<size_t>(port);
}

inline constexpr size_t paged_ssm_output_count = output_port_index(PagedSelectiveSSMOutputPort::Count);

inline constexpr std::array paged_ssm_computation_ports{
    PagedSelectiveSSMInputPort::A,
    PagedSelectiveSSMInputPort::TimeStep,
    PagedSelectiveSSMInputPort::InputProjection,
    PagedSelectiveSSMInputPort::Input,
    PagedSelectiveSSMInputPort::OutputProjection,
};

inline constexpr std::array paged_ssm_metadata_ports{
    PagedSelectiveSSMInputPort::SubsequenceBegins,
    PagedSelectiveSSMInputPort::BlockIndices,
    PagedSelectiveSSMInputPort::BlockIndicesBegins,
    PagedSelectiveSSMInputPort::NumProcessedTokens,
    PagedSelectiveSSMInputPort::CacheInterval,
};

inline bool is_paged_ssm_computation_port(PagedSelectiveSSMInputPort port) noexcept {
    return std::any_of(paged_ssm_computation_ports.begin(),
                       paged_ssm_computation_ports.end(),
                       [port](PagedSelectiveSSMInputPort computation_port) {
                           return port == computation_port;
                       });
}

inline bool is_paged_ssm_metadata_port(PagedSelectiveSSMInputPort port) noexcept {
    return std::any_of(paged_ssm_metadata_ports.begin(),
                       paged_ssm_metadata_ports.end(),
                       [port](PagedSelectiveSSMInputPort metadata_port) {
                           return port == metadata_port;
                       });
}

inline bool is_paged_ssm_float_port(PagedSelectiveSSMInputPort port) noexcept {
    return is_paged_ssm_computation_port(port) || port == PagedSelectiveSSMInputPort::State;
}

}  // namespace ov::intel_cpu
