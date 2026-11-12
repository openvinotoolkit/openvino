// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_reader_handler.hpp"

#include <algorithm>
#include <utility>

#include "hsm_common.hpp"
#include "hsm_writer_handler.hpp"
#include "openvino/core/coordinate.hpp"
#include "openvino/core/except.hpp"

namespace ov::hetero {

namespace {

// Stores a decoded section; a second section of the same kind makes the blob invalid.
template <class Slot, class T>
void hsm_store_unique(Slot& slot, T value, const char* section_name) {
    OPENVINO_ASSERT(!slot, "[HETERO] Cannot import compiled blob: duplicated ", section_name, " section.");
    slot = std::move(value);
}

// The only requirement HETERO has: it must understand the layout the blob was written with.
void hsm_check_requirements(const std::string& requirements) {
    OPENVINO_ASSERT(requirements == build_hsm_runtime_requirements(),
                    "[HETERO] Cannot import compiled blob: incompatible runtime requirements, blob: ",
                    requirements,
                    ", current: ",
                    build_hsm_runtime_requirements());
}

}  // namespace

std::string read_hsm_string(const runtime::hsm::Section& section) {
    std::string value(section.size(), '\0');
    OPENVINO_ASSERT(section.read(value.data()), "[HETERO] Cannot import compiled blob: failed to read a section.");
    return value;
}

ov::Tensor read_hsm_tensor(const HsmLocatedSection& located, const HsmBlobSource& source) {
    const auto size = located.section.size();
    if (size == 0) {
        return {};
    }

    const auto& parent = source.tensor;
    if (parent && parent.get_element_type() == ov::element::u8 && parent.get_shape().size() == 1) {
        const auto view = located.section.view();
        const auto base = static_cast<const std::byte*>(parent.data());
        if (!view.empty() && view.data() >= base && view.data() + view.size() <= base + parent.get_byte_size()) {
            const auto begin = static_cast<size_t>(view.data() - base);
            return ov::Tensor(parent, ov::Coordinate{begin}, ov::Coordinate{begin + view.size()});
        }
    }

    ov::Tensor tensor(ov::element::u8, ov::Shape{size});
    OPENVINO_ASSERT(located.section.read(tensor.data()),
                    "[HETERO] Cannot import compiled blob: failed to read a section.");
    return tensor;
}

void check_hsm_runtime_requirements(const runtime::hsm::Reader& reader) {
    const auto& id = hsm::runtime_requirements;
    const auto count = reader.count(id.device, id.tag.id());
    OPENVINO_ASSERT(count == 1,
                    "[HETERO] Cannot import compiled blob: expected one runtime requirements section, found ",
                    count);
    const auto section = reader.section(id.device, id.tag.id());
    OPENVINO_ASSERT(section->tag().is_pointer(), "[HETERO] Cannot import compiled blob: unexpected inline section.");
    hsm_check_requirements(read_hsm_string(*section));
}

bool HsmReaderHandler::handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) {
    const auto key = hsm::section_key(entry.device, entry.tag.id());
    const auto index = hsm::decode_section_index(entry.tag_reserved);
    const auto located = [&entry, &section] {
        OPENVINO_ASSERT(entry.tag.is_pointer(), "[HETERO] Cannot import compiled blob: unexpected inline section.");
        return HsmLocatedSection{entry, section};
    };

    if (key == hsm::section_key(hsm::metadata)) {
        hsm_store_unique(m_metadata, located(), "metadata");
    } else if (key == hsm::section_key(hsm::submodel_blob)) {
        hsm_store_unique(m_submodels[index].blob, located(), "submodel blob");
    } else if (key == hsm::section_key(hsm::submodel_ir_xml)) {
        hsm_store_unique(m_submodels[index].ir_xml, located(), "submodel IR xml");
    } else if (key == hsm::section_key(hsm::submodel_ir_weights)) {
        hsm_store_unique(m_submodels[index].ir_weights, located(), "submodel IR weights");
    } else {
        return false;
    }
    return true;
}

void HsmReaderHandler::check_structure() const {
    OPENVINO_ASSERT(m_metadata, "[HETERO] Cannot import compiled blob: missing metadata section.");
    size_t expected_index = 0;
    for (const auto& [index, submodel] : m_submodels) {
        // std::map is ordered: indexes must be exactly 0..n-1.
        OPENVINO_ASSERT(index == expected_index++,
                        "[HETERO] Cannot import compiled blob: unexpected submodel index ",
                        index);
        const auto has_blob = submodel.blob.has_value();
        const auto has_ir = submodel.ir_xml.has_value();
        // IR weights are the optional half of the IR variant and make no sense next to a blob.
        const auto is_single_variant = has_blob != has_ir && (has_ir || !submodel.ir_weights);
        OPENVINO_ASSERT(is_single_variant,
                        "[HETERO] Cannot import compiled blob: a submodel must have either a blob or IR sections, "
                        "submodel index: ",
                        index);
    }
}

}  // namespace ov::hetero
