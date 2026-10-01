// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/tensor_decryption.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/util/codec_xor.hpp"

namespace {

constexpr std::string_view COMPILER_SCHEDULE_CONTENT = "dummy";
constexpr std::string_view LOGGER_NAME = "dummy";

}  // namespace

using namespace intel_npu;

using testing::_;

using TensorDecryption = testing::Test;

TEST_F(TensorDecryption, ThrowsIfNullCallback) {
    ov::Tensor tensor(ov::element::Type_t::u8, {COMPILER_SCHEDULE_CONTENT.size()}, COMPILER_SCHEDULE_CONTENT.data());
    OV_EXPECT_THROW(utils::decrypt_payload(tensor, nullptr, Logger(LOGGER_NAME.data(), Logger::global().level())),
                    ov::Exception,
                    _);
}

TEST_F(TensorDecryption, WorkingScenario) {
    ov::Tensor tensor(ov::element::Type_t::u8, {COMPILER_SCHEDULE_CONTENT.size()}, COMPILER_SCHEDULE_CONTENT.data());
    const Logger logger(LOGGER_NAME.data(), Logger::global().level());

    utils::decrypt_payload(tensor, ov::util::codec_xor, logger);
    const std::string result(tensor.data<char>(), tensor.get_byte_size());
    const std::string reference = ov::util::codec_xor(std::string(COMPILER_SCHEDULE_CONTENT));

    // Take into account page alignment
    if (reference.size() % utils::STANDARD_PAGE_SIZE == 0) {
        ASSERT_EQ(result, reference);
    } else {
        ASSERT_EQ(result.size(), utils::align_size_to_standard_page_size(reference.size()));

        const std::string padding(result.size() - reference.size(), 0);
        ASSERT_EQ(result, reference + padding);
    }
}
