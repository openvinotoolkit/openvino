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

enum class BlobSourceDataType { STREAM, TENSOR };
const std::set<BlobSourceDataType> ALL_BLOB_SOURCE_DATA_TYPES({BlobSourceDataType::STREAM, BlobSourceDataType::TENSOR});

constexpr std::string_view TEST_STRING_STANDARD = "This is a test string";
constexpr std::string_view TEST_STRING_SPECIAL_CHARS = "This i\t a \ntest s\rtr!@#$%^()_+i&*ng";
constexpr std::string_view TEST_BUFFER = "\x00\x01\x02\x03";

constexpr std::string_view INVALID_BLOB_TYPE_MESSAGE = "Invalid blob type";

constexpr size_t DUMMY_BYTE = 0;

inline std::string get_content_type_name(const BlobContentType content_type) {
    switch (content_type) {
    case BlobContentType::STANDARD_STRING: {
        return "standard_string";
    }
    case BlobContentType::SPECIAL_CHARS_STRING: {
        return "special_chars_string";
    }
    case BlobContentType::BUFFER: {
        return "buffer";
    }
    default: {
        OPENVINO_THROW(INVALID_BLOB_TYPE_MESSAGE);
    }
    }
}

inline std::string get_source_data_type_name(const BlobSourceDataType source_data_type) {
    switch (source_data_type) {
    case BlobSourceDataType::STREAM: {
        return "stream";
    }
    case BlobSourceDataType::TENSOR: {
        return "tensor";
    }
    default: {
        OPENVINO_THROW(INVALID_BLOB_TYPE_MESSAGE);
    }
    }
}

}  // namespace

using testing::_;

class BlobSourceDifferentBlobs : public testing::Test,
                                 public testing::WithParamInterface<std::tuple<BlobContentType, BlobSourceDataType>> {
public:
    void SetUp() override {
        BlobContentType content_type;
        std::tie(content_type, source_data_type) = GetParam();

        switch (content_type) {
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

    static std::string getTestCaseName(
        const testing::TestParamInfo<std::tuple<BlobContentType, BlobSourceDataType>>& obj) {
        BlobContentType content_type;
        BlobSourceDataType source_data_type;
        std::tie(content_type, source_data_type) = GetParam();

        return get_content_type_name(content_type) + "_" + get_source_data_type_name(source_data_type);
    }

protected:
    BlobSource create_blob_source() {
        switch (source_data_type) {
        case BlobSourceDataType::STREAM: {
            return BlobSource(stream);
        }
        case BlobSourceDataType::TENSOR: {
            return BlobSource(tensor);
        }
        default: {
            OPENVINO_THROW("Invalid blob source data type");
        }
        }
    }

    std::string_view blob_content;
    BlobSourceDataType source_data_type;
    std::istringstream stream;
    ov::Tensor tensor;
};

/**
 * @brief
 */
TEST_P(BlobSourceDifferentBlobs, CopyFirstByte) {
    BlobSource blob_source = create_blob_source();

    const size_t copy_size = 1;
    std::string copied_payload(copy_size, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(0, copy_size));

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, copy_size);
}

INSTANTIATE_TEST_SUITE_P(UnitTest,
                         BlobSourceDifferentBlobs,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::ValuesIn(ALL_BLOB_SOURCE_DATA_TYPES)),
                         BlobSourceDifferentBlobs::getTestCaseName);
