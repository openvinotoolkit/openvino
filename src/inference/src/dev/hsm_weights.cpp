// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_weights.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
namespace {

class WeightsEncoder final : public ISectionEncoder {
public:
    explicit WeightsEncoder(ov::util::MemoryView weights) : m_weights(weights) {}

    void encode(const SectionSink& sink) const override {
        sink(m_weights);
    }

private:
    ov::util::MemoryView m_weights;
};

}  // namespace

SectionEncoderPtr encode_weights(ov::util::MemoryView weights) {
    return std::make_shared<WeightsEncoder>(weights);
}

std::optional<WeightsBuffer> decode_weights(const Section& section) noexcept try {
    if (section.size() == 0) {
        return WeightsBuffer{};
    }
    return section.to_bytes();
} catch (...) {
    return std::nullopt;
}

std::optional<ov::util::MemoryView> decode_weights_view(const Section& section) noexcept {
    if (!section.tag().is_pointer()) {
        return std::nullopt;
    }

    if (section.size() == 0) {
        return ov::util::MemoryView{};
    }

    const auto view = section.view();
    return view.empty() ? std::nullopt : std::optional{view};
}

}  // namespace v1
}  // namespace ov::runtime::hsm
