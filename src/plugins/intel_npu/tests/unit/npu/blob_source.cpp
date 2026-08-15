// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_source.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <set>
#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"

using namespace intel_npu;

namespace {

enum class BlobContentType { STANDARD_STRING, SPECIAL_CHARS_STRING, BUFFER };
const std::set<BlobContentType> ALL_BLOB_CONTENT_TYPES({BlobContentType::STANDARD_STRING,
                                                        BlobContentType::SPECIAL_CHARS_STRING,
                                                        BlobContentType::BUFFER});

constexpr std::string_view TEST_STRING_STANDARD_NAME = "This is a test string";
constexpr std::string_view TEST_STRING_SPECIAL_CHARS_NAME = "This i\t a \ntest s\rtr!@#$%^()_+i&*ng";
constexpr std::string_view TEST_BUFFER_NAME = "\x00\x01\x02\x03";

constexpr std::string_view TEST_STRING_STANDARD = "This is a test string";
constexpr std::string_view TEST_STRING_SPECIAL_CHARS = "This i\t a \ntest s\rtr!@#$%^()_+i&*ng";
constexpr std::string_view TEST_BUFFER = "\x00\x01\x02\x03";

constexpr std::string_view INVALID_BLOB_TYPE_MESSAGE = "Invalid blob type";

}  // namespace

using testing::_;

class BlobSourceDifferentBlobs : public testing::Test, public testing::WithParamInterface<BlobContentType> {
public:
    void SetUp() override {
        switch (GetParam()) {
        case BlobContentType::STANDARD_STRING: {
            blob_content = TEST_STRING_STANDARD;
            break;
        }
        case BlobContentType::SPECIAL_CHARS_STRING: {
            blob_content = TEST_STRING_SPECIAL_CHARS;
            break;
        }
        case BlobContentType::BUFFER: {
            blob_content = TEST_BUFFER;
            break;
        }
        default: {
            OPENVINO_THROW(INVALID_BLOB_TYPE_MESSAGE);
        }
        }

        stream = std::istringstream(blob_content.data());
        tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape({blob_content.size()}), blob_content.data());
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BlobContentType>& obj) {
        switch (obj.param) {
        case BlobContentType::STANDARD_STRING: {
            return TEST_STRING_STANDARD_NAME.data();
        }
        case BlobContentType::SPECIAL_CHARS_STRING: {
            return TEST_STRING_SPECIAL_CHARS_NAME.data();
        }
        case BlobContentType::BUFFER: {
            return TEST_BUFFER_NAME.data();
        }
        default: {
            OPENVINO_THROW(INVALID_BLOB_TYPE_MESSAGE);
        }
        }
    }

protected:
    std::string_view blob_content;
    std::istringstream stream;
    ov::Tensor tensor;
};

/**
 * @brief
 */
TEST_P(BlobSourceDifferentBlobs, ReadFirstByte) {
    BlobSource stream_blob_source(stream);
    BlobSource tensor_blob_source(tensor);
}

INSTANTIATE_TEST_SUITE_P(UnitTest,
                         BlobSourceDifferentBlobs,
                         testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                         BlobSourceDifferentBlobs::getTestCaseName);
