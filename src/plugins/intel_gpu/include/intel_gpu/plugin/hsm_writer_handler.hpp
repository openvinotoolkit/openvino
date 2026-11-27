// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <ostream>
#include <string>

#include "intel_gpu/plugin/hsm_common.hpp"
#include "openvino/runtime/hsm_writer.hpp"
#include "openvino/runtime/properties.hpp"

namespace cldnn {
class BinaryOutputBuffer;
}  // namespace cldnn

namespace ov::intel_gpu::hsm {

class WriterHandler final : public runtime::hsm::ISectionWriterHandler {
public:
    WriterHandler(ov::CacheMode cache_mode,
                  std::string requirements,
                  std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                  const decltype(ov::EncryptionCallbacks::encrypt)& encrypt,
                  size_t plaintext_capacity);

    void handle_section(runtime::hsm::IWriter& writer) const override;

private:
    PayloadInfo m_info;
    std::string m_requirements;
    ov::AnyMap m_compile_options;
    runtime::hsm::SectionEncoderPtr m_encoder;
};

/// Writes the HSM container; @p serialize streams the model payload into the model section.
/// With @p encrypt set, the whole plaintext payload and its ciphertext are held in memory (about 2x the payload),
/// because the callback takes and returns whole strings.
void write_model(std::ostream& destination,
                 ov::CacheMode cache_mode,
                 const std::string& requirements,
                 std::function<void(cldnn::BinaryOutputBuffer&)> serialize,
                 const decltype(ov::EncryptionCallbacks::encrypt)& encrypt,
                 size_t plaintext_capacity);

}  // namespace ov::intel_gpu::hsm
