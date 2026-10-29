// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_weights.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

bool encode_weights(IWriter& writer, ov::util::MemoryView weights, SectionAlignment align) {
    return writer.add_section(any_device_id, weights_tag, weights, align);
}

std::optional<WeightsBuffer> decode_weights(const Reader& reader) {
    const SectionDecoder<WeightsBuffer> decoder = [](const Section& section) {
        return section.to_bytes();
    };
    return reader.decode(Tags::weights, decoder);
}

std::optional<ov::util::MemoryView> decode_weights_view(const Reader& reader) {
    const auto section = reader.section(Tags::weights);
    if (!section || !section->tag().is_pointer()) {
        return std::nullopt;
    }

    if (section->size() == 0) {
        return ov::util::MemoryView{};
    }

    const auto view = section->view();
    return view.empty() ? std::nullopt : std::optional{view};
}

}  // namespace v1
}  // namespace ov::runtime::hsm
