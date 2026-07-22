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
         make_key_value_field(MetadataTextKeys::WS_INITS, "1"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.2") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::BATCH, "4"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.2") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::BATCH, "4") + ";" +
         make_key_value_field(MetadataTextKeys::WS_INITS, "1"),
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.2") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::BATCH, "-1"),
     false},
    // order independent: metadata version field is not first
    {make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" + make_key_value_field(MetadataTextKeys::META, "2.0"),
     true},
    // optional (unknown) fields are ignored
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";{future_field=FUTURE_VAL}",
     true},
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";future_field=FUTURE_VAL",
     false},
    {"", false},
    {make_key_value_field(MetadataTextKeys::OV, "2026.1.0"), false},
    // metadata version malformed
    {make_key_value_field(MetadataTextKeys::META, "20") + ";" + make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2X.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.0X") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.0.1") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0"),
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
    {make_key_value_field(MetadataTextKeys::META, "2.1") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::WS_INITS, "0"),
     false},
    // compatibility descriptor value is not bracket-enclosed
    {make_key_value_field(MetadataTextKeys::META, "2.6") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";" +
         make_key_value_field(MetadataTextKeys::COMPAT_DESC, "platform=NPU3720"),
     false},
    // unmatched closing bracket in value
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";bad=]VALUE",
     false},
    // unclosed opening bracket in value
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";bad=[VALUE",
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" + make_key_value_field(MetadataTextKeys::OV, "2026.1"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" + make_key_value_field(MetadataTextKeys::OV, "20261"),
     false},
    // too many ov version components
    {make_key_value_field(MetadataTextKeys::META, "2.0") + ";" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0.0"),
     false},
    {make_key_value_field(MetadataTextKeys::META, "2.6") + ";;" +
         make_key_value_field(MetadataTextKeys::OV, "2026.1.0") + ";",
     false},
    {"key=VALUE;key=VALUE2", false},
    {"notavalidformat", false},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(FormatValidation,
                         MetadataTextTest,
                         ::testing::ValuesIn(inputs),
                         MetadataTextTest::getTestCaseName);
