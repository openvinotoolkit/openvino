// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/compatibility_string.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"

using namespace intel_npu;

using StringCompatibilityEncodingTests = ::testing::Test;

TEST_F(StringCompatibilityEncodingTests, sameStringAfterEncodeDecode) {
    const std::vector<uint8_t> input_sequence{'a', 'b', '\0', 1, 2, 'c'};
    std::ostringstream input_stringstream;
    input_stringstream.write(reinterpret_cast<const char*>(input_sequence.data()),
                             static_cast<std::streamsize>(input_sequence.size()));

    const std::string input_string = input_stringstream.str();
    const std::string encoded_string = utils::encode_compatibility_string(input_string);
    const std::string reference = "616200010263";
    ASSERT_EQ(encoded_string, reference);

    const std::string decoded_string = utils::decode_compatibility_string(encoded_string);
    ASSERT_EQ(decoded_string, input_string);
}
