// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata_text.hpp"

namespace {

const auto make_key_value_field = [](std::string_view key, std::string_view val) -> std::string {
    return std::string(key) + '=' + std::string(val);
};

const std::vector<MetadataTextTest::ParamType> inputs = {
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" + make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.1") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::WS_INITS, "[128]"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.1") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::WS_INITS, "[16|32|64]"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.2") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::BATCH, "4"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.6") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::COMPILER_REQS, "[platform=NPU3720;tiles=2]"),
     true},
    // order independent: metadata version field is not first
    {make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" + make_key_value_field(MetadataTextKeys::META, "2.0"),
     true},
    // unknown fields are ignored
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";future_field=some_value",
     true},
    // invalid cases
    {"", false},
    {make_key_value_field(MetadataTextKeys::OV, "2026.1.0"), false},
    // metadata version malformed
    {make_key_value_field(MetadataTextKeys::META, "20") + ";" + make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    // unsupported metadata major version
    {make_key_value_field(MetadataTextKeys::META, std::to_string(CURRENT_METADATA_MAJOR_VERSION + 1) + ".0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    // minor version exceeds current maximum
    {make_key_value_field(MetadataTextKeys::META, "2.99") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.0"), false},
    // list missing brackets
    {make_key_value_field(MetadataTextKeys::META, "2.1") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::WS_INITS, "16|32|64"),
     false},
    // compiler_reqs value is not bracket-enclosed
    {make_key_value_field(MetadataTextKeys::META, "2.6") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::COMPILER_REQS, "platform=NPU3720"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.6") + ";;" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";",
     false},
    {"key=value;key=;", false},
    {"notavalidformat", false},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(FormatValidation,
                         MetadataTextTest,
                         ::testing::ValuesIn(inputs),
                         MetadataTextTest::getTestCaseName);
