// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>
#include <optional>
#include <streambuf>
#include <string>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace cldnn {
class BinaryInputBuffer;
class engine;
}  // namespace cldnn

namespace ov::intel_gpu::hsm {

/// Reads the GPU sections of an HSM container and opens its model section for graph reconstruction.
class ReaderHandler final : public runtime::hsm::ISectionReaderHandler {
public:
    /// Collects and validates the sections, and rejects incompatible runtime requirements before the payload is read.
    ReaderHandler(const runtime::hsm::Reader& reader, cldnn::engine& engine);
    ~ReaderHandler() override;

    bool handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) override;

    ov::CacheMode cache_mode() const;

    /// Model payload read from @p source, in which the container starts at @p container_start.
    cldnn::BinaryInputBuffer& open_model(std::istream& source,
                                         std::streampos container_start,
                                         const ov::EncryptionCallbacks& callbacks);

    /// Model payload read in place from @p blob, the memory the reader was opened over.
    cldnn::BinaryInputBuffer& open_model(const ov::Tensor& blob, const ov::EncryptionCallbacks& callbacks);

private:
    bool open_encrypted_model(const ov::EncryptionCallbacks& callbacks);

    cldnn::engine& m_engine;
    std::optional<PayloadInfo> m_info;
    std::optional<std::string> m_requirements;
    std::optional<ov::CacheMode> m_cache_mode;
    std::optional<runtime::hsm::Section> m_model;
    runtime::hsm::OffsetType m_model_offset = 0;
    std::string m_plaintext;
    ov::Tensor m_model_tensor;
    std::unique_ptr<std::streambuf> m_buffer;
    std::unique_ptr<std::istream> m_stream;
    std::unique_ptr<cldnn::BinaryInputBuffer> m_input;
};

}  // namespace ov::intel_gpu::hsm
