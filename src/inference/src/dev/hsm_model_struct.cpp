// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
void encode_model_struct(std::string_view xml, const SectionSink& sink) {
    sink({reinterpret_cast<const std::byte*>(xml.data()), xml.size()});
}

std::optional<std::string> decode_model_struct(ov::util::MemoryView payload) {
    if (payload.size() && payload.empty()) {
        return std::nullopt;
    }
    return payload.size() ? std::optional<std::string>{{reinterpret_cast<const char*>(payload.data()), payload.size()}}
                          : std::optional<std::string>{std::string{}};
}

std::optional<std::string> decode_model_struct(const Section& section) noexcept try {
    if (section.size() == 0) {
        return std::string{};
    }
    if (const auto view = section.view(); !view.empty()) {
        return decode_model_struct(view);
    }
    std::string xml(section.size(), '\0');
    return section.read(xml.data()) ? std::optional{std::move(xml)} : std::nullopt;
} catch (...) {
    return std::nullopt;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
