// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <map>
#include <optional>
#include <string>

#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::hetero {

/// Where the sections of an opened container live, for sections that are imported by the sub-devices.
struct HsmBlobSource {
    std::istream* stream = nullptr;  //!< Set when the container is read from a stream.
    std::streampos start{};          //!< Container start in #stream.
    ov::Tensor tensor;               //!< Set when the container is read from a buffer.
};

/// A section together with its manifest entry, which holds the offset needed to address it in a stream.
struct HsmLocatedSection {
    runtime::hsm::ManifestEntry entry;
    runtime::hsm::Section section;
};

class HsmReaderHandler final : public runtime::hsm::ISectionReaderHandler {
public:
    struct Submodel {
        std::optional<HsmLocatedSection> blob;
        std::optional<HsmLocatedSection> ir_xml;
        std::optional<HsmLocatedSection> ir_weights;
    };

    bool handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) override;

    /// Validates the sections found, without reading any of them: the metadata is present and the submodels are
    /// numbered 0..n-1, each with either a blob or IR sections.
    void check_structure() const;

    const HsmLocatedSection& metadata() const {
        return *m_metadata;
    }

    size_t submodel_count() const {
        return m_submodels.size();
    }

    const Submodel& submodel(size_t index) const {
        return m_submodels.at(index);
    }

private:
    std::optional<HsmLocatedSection> m_metadata;
    std::map<size_t, Submodel> m_submodels;
};

/// Throws unless the container has exactly one runtime requirements section that this HETERO can import.
/// Call it before reading any other section.
void check_hsm_runtime_requirements(const runtime::hsm::Reader& reader);

std::string read_hsm_string(const runtime::hsm::Section& section);

/// Reads the section into a tensor; maps the bytes of the source tensor instead of copying when possible.
ov::Tensor read_hsm_tensor(const HsmLocatedSection& located, const HsmBlobSource& source);

}  // namespace ov::hetero
