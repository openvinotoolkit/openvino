// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include <utility>

namespace ov::runtime::hsm {
inline namespace v1 {
namespace {

class ModelStructEncoder final : public ISectionEncoder {
public:
    explicit ModelStructEncoder(std::string xml) : m_xml(std::move(xml)) {}

    void encode(const SectionSink& sink) const override {
        sink({reinterpret_cast<const std::byte*>(m_xml.data()), m_xml.size()});
    }

private:
    std::string m_xml;
};

}  // namespace

SectionEncoderPtr encode_model_struct(std::string xml) {
    return std::make_shared<ModelStructEncoder>(std::move(xml));
}

std::optional<std::string> decode_model_struct(const Section& section) noexcept try {
    std::string xml(section.size(), '\0');
    return section.read(xml.data()) ? std::optional{std::move(xml)} : std::nullopt;
} catch (...) {
    return std::nullopt;
}

std::optional<std::string_view> decode_model_struct_view(const Section& section) noexcept {
    if (!section.tag().is_pointer()) {
        return std::nullopt;
    }

    if (section.size() == 0) {
        return std::string_view{};
    }

    const auto view = section.view();
    return view.empty() ? std::nullopt
                        : std::optional<std::string_view>{{reinterpret_cast<const char*>(view.data()), view.size()}};
}

}  // namespace v1
}  // namespace ov::runtime::hsm
