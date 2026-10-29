// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "util.hpp"

namespace {

TEST(NPUWRestoredKVNames, MapsOutputsBackToInputs) {
    for (const std::string kind : {"key", "value"}) {
        const auto input_name = "input_restored.past_key_values.0.decoder." + kind;
        const auto output_name = "output_restored.past_key_values.0.decoder." + kind;
        EXPECT_EQ(ov::npuw::util::past_key_values_to_present_name(input_name), output_name);
        EXPECT_EQ(ov::npuw::util::present_to_past_key_values_name(output_name), input_name);
        const auto resolved = ov::npuw::util::resolveKVInputName(output_name, [&](const std::string& name) {
            return name == input_name;
        });
        ASSERT_TRUE(resolved.has_value());
        EXPECT_EQ(resolved.value(), input_name);
        EXPECT_TRUE(ov::npuw::util::isKVCacheName(input_name));
        EXPECT_TRUE(ov::npuw::util::isKVCacheName(output_name));
        EXPECT_EQ(ov::npuw::util::isPastValueParam(input_name), kind == "value");
    }
    EXPECT_FALSE(ov::npuw::util::isKVCacheName("input_restored.encoder_hidden_states"));
}

TEST(NPUWRestoredKVNames, PreservesCanonicalAndQuantizedNames) {
    EXPECT_EQ(ov::npuw::util::past_key_values_to_present_name("past_key_values.0.key"), "present.0.key");
    EXPECT_EQ(ov::npuw::util::present_to_past_key_values_name("present.0.value"), "past_key_values.0.value");
    EXPECT_EQ(ov::npuw::util::past_key_values_to_present_name("DynamicQuantize/0/past_key_values/value/scale"),
              "DynamicQuantize/0/present/value/scale");
    EXPECT_TRUE(ov::npuw::util::isDQScaleOrZPValue("DynamicQuantize/0/past_key_values/value/scale"));
    EXPECT_FALSE(ov::npuw::util::isPastValueParam("past_key_values.0.key"));
    EXPECT_TRUE(ov::npuw::util::isPastValueParam("past_key_values.0.value"));
}

}  // namespace
