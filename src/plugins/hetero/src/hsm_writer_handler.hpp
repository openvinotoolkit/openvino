// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::hetero {

/// Content of one submodel: a blob written by the sub-device, or an IR fallback.
struct HsmSubmodelPayload {
    std::function<void(std::ostream&)> export_blob;  //!< Set for a blob; the stream is seekable.
    std::shared_ptr<ov::Model> ir_model;  //!< Serialized straight into the sections, when there is no #export_blob.
};

std::string build_hsm_runtime_requirements();

class HsmWriterHandler final : public runtime::hsm::ISectionWriterHandler {
public:
    /// @p destination is the stream the IWriter writes into; it must be seekable.
    HsmWriterHandler(std::ostream& destination, std::string metadata, std::vector<HsmSubmodelPayload> submodels);

    /// The handler must outlive IWriter::finalize(): the sections borrow its buffers and run its exporters.
    void handle_section(runtime::hsm::IWriter& writer) const override;

private:
    std::ostream& m_destination;
    std::string m_runtime_requirements;
    std::string m_metadata;
    std::vector<HsmSubmodelPayload> m_submodels;
};

}  // namespace ov::hetero
