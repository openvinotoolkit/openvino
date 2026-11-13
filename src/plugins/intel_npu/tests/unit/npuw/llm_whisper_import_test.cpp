// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Export/import regression tests for the NPUW Whisper pipeline.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>

#include "llm_test_helpers.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

using ov::test::npuw::build_whisper_decoder_test_model;

class LLMWhisperImportTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_dir = std::filesystem::temp_directory_path() / "npuw_whisper_import";
        std::filesystem::create_directories(m_dir);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(m_dir, ec);
    }

    std::filesystem::path m_dir;
};

// The EOS token is baked into the compiled Whisper model. After import, the reported property must still match
// the one inference uses, even though NPUW_WHISPER_EOS_TOKEN is not passed to import_model.
TEST_F(LLMWhisperImportTest, ImportedModelReportsCompiledEosToken) {
    constexpr uint64_t kCustomEosToken = 42;  // differs from the option default
    static_assert(kCustomEosToken != 50257, "the test needs a non-default EOS token");

    ov::Core core;
    const auto xml = m_dir / "model.xml";
    const auto bin = m_dir / "model.bin";
    ov::serialize(build_whisper_decoder_test_model(), xml.string(), bin.string());
    const auto model = core.read_model(xml.string(), bin.string());

    // NPUW_ONLINE_PIPELINE=NONE because of the synthetic model
    const ov::AnyMap props{{"NPU_USE_NPUW", "YES"},
                           {"NPUW_LLM", "YES"},
                           {"NPUW_WHISPER", "YES"},
                           {"NPUW_WHISPER_EOS_TOKEN", std::to_string(kCustomEosToken)},
                           {"NPUW_DEVICES", "CPU"},
                           {"NPUW_ONLINE_PIPELINE", "NONE"},
                           {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                           {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    auto compiled = core.compile_model(model, "NPU", props);
    ASSERT_EQ(compiled.get_property("NPUW_WHISPER_EOS_TOKEN").as<uint64_t>(), kCustomEosToken);

    std::stringstream blob;
    compiled.export_model(blob);

    // Import without passing the EOS token: the value must come from the blob
    const ov::AnyMap import_props{{"NPU_USE_NPUW", "YES"}, {ov::weights_path.name(), bin.string()}};
    auto imported = core.import_model(blob, "NPU", import_props);

    EXPECT_EQ(imported.get_property("NPUW_WHISPER_EOS_TOKEN").as<uint64_t>(), kCustomEosToken);
}

}  // namespace
