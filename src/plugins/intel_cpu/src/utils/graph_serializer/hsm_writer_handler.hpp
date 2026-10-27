// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "utils/graph_serializer/serializer.hpp"

namespace ov::intel_cpu {

class HsmWriterHandler final : public runtime::hsm::ISectionWriterHandler {
public:
    HsmWriterHandler(std::shared_ptr<ov::Model> model,
                     std::string runtime_requirements,
                     ModelSerializer::CacheEncrypt encrypt_fn,
                     bool weightless_mode);

    /// The handler must outlive IWriter::finalize(): the model is serialized straight into the writer at that point.
    void handle_section(runtime::hsm::IWriter& writer) const override;

private:
    std::string m_runtime_requirements;
    runtime::hsm::SectionEncoderPtr m_weights_encoder;
    runtime::hsm::SectionEncoderPtr m_model_encoder;
};

}  // namespace ov::intel_cpu
