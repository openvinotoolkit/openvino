// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_id_codec.hpp"

#include <cassert>

#include "openvino/core/except.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

std::optional<uint64_t> decode_model_id(const Section& section) noexcept {
    uint64_t id{};
    try {
        return section.read(0, sizeof(id), &id) ? std::optional{id} : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
