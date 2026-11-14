// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_id_codec.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

// Once CVS-198883 resolves noexcept assurance issue, coverity warning shall be unsilenced
// coverity[UNCAUGHT_EXCEPT:FALSE]
std::optional<uint64_t> decode_model_id(const Section& section) noexcept {
    uint64_t id{};
    return section.read(0, sizeof(id), &id) ? std::optional{id} : std::nullopt;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
