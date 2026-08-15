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

class BlobSourceDifferentBlobs : public testing::TestWithParam<std::tuple<BlobContentType, BlobSourceDataType>> {
public:
    static std::string getTestCaseName(
        const testing::TestParamInfo<std::tuple<BlobContentType, BlobSourceDataType>>& obj) {
        BlobContentType content_type;
        BlobSourceDataType source_data_type;
        std::tie(content_type, source_data_type) = GetParam();

        return get_content_type_name(content_type) + "_" + get_source_data_type_name(source_data_type);
    }

protected:
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

TEST_P(BlobSourceDifferentBlobs, CopyAllBytes) {
    BlobSource blob_source = create_blob_source();

    std::string copied_payload(blob_content.size(), DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorToStartReferenceBeginning) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(0, std::ios::beg);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 0);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(0, 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorToLastByteReferenceBeginning) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(blob_content.size() - 1, std::ios::beg);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(blob_content.size() - 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorToStartReferenceEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(0);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor(), std::ios::end);
    ASSERT_EQ(cursor, 0);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(0, 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorToLastByteReferenceEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(blob_content.size() - 1, std::ios::end);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(blob_content.size() - 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorTwiceForward) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(1, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 1);

    blob_source.move_cursor(1, std::ios::cur);
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 2);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(2, 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 3);
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorTwiceBackwards) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(0, std::ios::end);
    blob_source.move_cursor(-1, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    blob_source.move_cursor(-1, std::ios::cur);
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size() - 2);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(blob_content.size() - 2, 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, blob_content.size() - 1);
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorSamePlaceReferenceCurrent) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(1, std::ios::beg);
    blob_source.move_cursor(0, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 1);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content.substr(1, 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.get_cursor());
    ASSERT_EQ(cursor, 2);
}

TEST_P(BlobSourceDifferentBlobs, GetRemainingSizeFromStart) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(blob_content.size() - 1);

    size_t remaining_size = 0;
    OV_ASSERT_NO_THROW(remaining_size = blob_source.get_remaining_size());
    ASSERT_EQ(remaining_size, blob_content.size());
}

TEST_P(BlobSourceDifferentBlobs, GetRemainingSizeFromEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(blob_content.size());

    size_t remaining_size = 0;
    OV_ASSERT_NO_THROW(remaining_size = blob_source.get_remaining_size());
    ASSERT_EQ(remaining_size, 0);
}

TEST_P(BlobSourceDifferentBlobs, GetTotalSize) {
    BlobSource blob_source = create_blob_source();

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_source.size());
}

TEST_P(BlobSourceDifferentBlobs, GetTotalSizeAfterCursorMove) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(blob_content.size());

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_source.size());
}

TEST_P(BlobSourceDifferentBlobs, GetTotalSizeAfterRead) {
    BlobSource blob_source = create_blob_source();

    std::string copied_payload(1, DUMMY_BYTE);
    OV_ASSERT_NO_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()));

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_source.size());
}

TEST_P(BlobSourceDifferentBlobs, CopyTooMuch) {
    BlobSource blob_source = create_blob_source();

    std::string copied_payload(blob_content.size() + 1, DUMMY_BYTE);
    OV_EXPECT_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()), ov::Exception, _);
}

TEST_P(BlobSourceDifferentBlobs, CopyAfterEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.move_cursor(0, std::ios::end);

    std::string copied_payload(1, DUMMY_BYTE);
    OV_EXPECT_THROW(blob_source.copy_from_source(copied_payload.data(), copied_payload.size()), ov::Exception, _);
}

TEST_P(BlobSourceDifferentBlobs, MoveCursorBeforeStartReferenceStart) {}

TEST_P(BlobSourceDifferentBlobs, MoveCursorAfterEndReferenceStart) {}

TEST_P(BlobSourceDifferentBlobs, MoveCursorBeforeStartReferenceCurrent) {}

TEST_P(BlobSourceDifferentBlobs, MoveCursorAfterEndReferenceCurrnet) {}

TEST_P(BlobSourceDifferentBlobs, MoveCursorBeforeStartReferenceEnd) {}

TEST_P(BlobSourceDifferentBlobs, MoveCursorAfterEndReferenceEnd) {}

// interpret errors for stream; same tests as copy for tensor

INSTANTIATE_TEST_SUITE_P(AllDataTypes,
                         BlobSourceDifferentBlobs,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::ValuesIn(ALL_BLOB_SOURCE_DATA_TYPES)),
                         BlobSourceDifferentBlobs::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(StreamDataType,
                         BlobSourceDifferentBlobs,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::ValuesIn(ALL_BLOB_SOURCE_DATA_TYPES)),
                         BlobSourceDifferentBlobs::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TensorDataType,
                         BlobSourceDifferentBlobs,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::ValuesIn(ALL_BLOB_SOURCE_DATA_TYPES)),
                         BlobSourceDifferentBlobs::getTestCaseName);
