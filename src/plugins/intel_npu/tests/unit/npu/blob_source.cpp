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
#include "intel_npu/utils/utils.hpp"

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
const std::vector<uint8_t> TEST_BUFFER{0, 1, 2, 3};

constexpr std::string_view INVALID_BLOB_TYPE_MESSAGE = "Invalid blob type";

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
        std::tie(content_type, source_data_type) = obj.param;

        return get_content_type_name(content_type) + "_" + get_source_data_type_name(source_data_type);
    }

protected:
    void SetUp() override {
        BlobContentType content_type;
        std::tie(content_type, source_data_type) = GetParam();

        switch (content_type) {
        case BlobContentType::STANDARD_STRING: {
            blob_content = std::vector<uint8_t>(TEST_STRING_STANDARD.begin(), TEST_STRING_STANDARD.end());
            break;
        }
        case BlobContentType::SPECIAL_CHARS_STRING: {
            blob_content = std::vector<uint8_t>(TEST_STRING_SPECIAL_CHARS.begin(), TEST_STRING_SPECIAL_CHARS.end());
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
    }

    BlobSource create_blob_source() {
        switch (source_data_type) {
        case BlobSourceDataType::STREAM: {
            stream = std::istringstream(std::string(blob_content.begin(), blob_content.end()));
            return BlobSource(stream);
        }
        case BlobSourceDataType::TENSOR: {
            tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape({blob_content.size()}), blob_content.data());
            return BlobSource(tensor);
        }
        default: {
            OPENVINO_THROW("Invalid blob source data type");
        }
        }
    }

    std::vector<uint8_t> blob_content;
    BlobSourceDataType source_data_type;
    std::istringstream stream;
    ov::Tensor tensor;
};

using BlobSourceDifferentBlobsCommon = BlobSourceDifferentBlobs;

// The data types can be contiguous (tensor) or non-contiguous (stream). Some functions behave differently based on
// this.
using BlobSourceDifferentBlobsNonContiguous = BlobSourceDifferentBlobs;
using BlobSourceDifferentBlobsContiguous = BlobSourceDifferentBlobs;

/**
 * @brief The first byte from the blob source can be copied correctly
 */
TEST_P(BlobSourceDifferentBlobsCommon, CopyFirstByte) {
    BlobSource blob_source = create_blob_source();

    const size_t copy_size = 1;
    std::vector<uint8_t> copied_payload(copy_size);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.begin(), blob_content.begin() + copy_size));

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, copy_size);
}

/**
 * @brief All bytes from the blob source can be copied correctly
 */
TEST_P(BlobSourceDifferentBlobsCommon, CopyAllBytes) {
    BlobSource blob_source = create_blob_source();

    std::vector<uint8_t> copied_payload(blob_content.size());
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, blob_content);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size());
}

/**
 * @brief The read cursor can be moved to the beginning of the source when using the start as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorToStartReferenceBeginning) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(0, std::ios::beg);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 0);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.begin(), blob_content.begin() + 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, copied_payload.size());
}

/**
 * @brief The read cursor can be moved to the last byte of the source when using the start as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorToLastByteReferenceBeginning) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(blob_content.size() - 1, std::ios::beg);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.end() - 1, blob_content.end()));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size());
}

/**
 * @brief The read cursor can be moved to the begining of the source when using the end as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorToStartReferenceEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(-static_cast<int64_t>(blob_content.size()), std::ios::end);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 0);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.begin(), blob_content.begin() + 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, copied_payload.size());
}

/**
 * @brief The read cursor can be moved to the last byte of the source when using the end as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorToLastByteReferenceEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(-1, std::ios::end);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.end() - 1, blob_content.end()));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size());
}

/**
 * @brief The read cursor can be moved twice forward when using the current position as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorTwiceForward) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(1, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 1);

    blob_source.seekg(1, std::ios::cur);
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 2);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.begin() + 2, blob_content.begin() + 3));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 3);
}

/**
 * @brief The read cursor can be moved twice backwards when using the current position as reference
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorTwiceBackwards) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(0, std::ios::end);
    blob_source.seekg(-1, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size() - 1);

    blob_source.seekg(-1, std::ios::cur);
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size() - 2);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.end() - 2, blob_content.end() - 1));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size() - 1);
}

/**
 * @brief The read cursor will stay in the previous spot when moved by 0 bytes
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorSamePlaceReferenceCurrent) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(1, std::ios::beg);
    blob_source.seekg(0, std::ios::cur);
    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 1);

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));
    ASSERT_EQ(copied_payload, std::vector<uint8_t>(blob_content.begin() + 1, blob_content.begin() + 2));

    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, 2);
}

/**
 * @brief The remaining size should be equal to the size of the source when the cursor is at origin
 */
TEST_P(BlobSourceDifferentBlobsCommon, GetRemainingSizeFromStart) {
    BlobSource blob_source = create_blob_source();

    size_t remaining_size = 0;
    OV_ASSERT_NO_THROW(remaining_size = blob_source.get_remaining_size());
    ASSERT_EQ(remaining_size, blob_content.size());
}

/**
 * @brief No remaining bytes should be reported is the cursor is at the end of the source
 */
TEST_P(BlobSourceDifferentBlobsCommon, GetRemainingSizeFromEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(blob_content.size());

    size_t remaining_size = 0;
    OV_ASSERT_NO_THROW(remaining_size = blob_source.get_remaining_size());
    ASSERT_EQ(remaining_size, 0);
}

/**
 * @brief The total size of the blob source is the same as the size of the underlying buffer
 */
TEST_P(BlobSourceDifferentBlobsCommon, GetTotalSize) {
    BlobSource blob_source = create_blob_source();

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_content.size());
}

/**
 * @brief The total size of the blob source is not affected by the position of the cursor
 */
TEST_P(BlobSourceDifferentBlobsCommon, GetTotalSizeAfterCursorMove) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(blob_content.size());

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_content.size());
}

/**
 * @brief The total size of the source should not be influenced by previous "read" operations
 */
TEST_P(BlobSourceDifferentBlobsCommon, GetTotalSizeAfterRead) {
    BlobSource blob_source = create_blob_source();

    std::vector<uint8_t> copied_payload(1);
    OV_ASSERT_NO_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()));

    size_t size = 0;
    OV_ASSERT_NO_THROW(size = blob_source.get_total_size());
    ASSERT_EQ(size, blob_content.size());
}

/**
 * @brief Requesting to copy more bytes than available should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsCommon, CopyTooMuch) {
    BlobSource blob_source = create_blob_source();

    std::vector<uint8_t> copied_payload(blob_content.size() + 1);
    OV_EXPECT_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()), ov::Exception, _);
}

/**
 * @brief Requesting to copy beyond the end limit should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsCommon, CopyAfterEnd) {
    BlobSource blob_source = create_blob_source();

    blob_source.seekg(0, std::ios::end);

    std::vector<uint8_t> copied_payload(1);
    OV_EXPECT_THROW(blob_source.read_into_buffer(copied_payload.data(), copied_payload.size()), ov::Exception, _);
}

/**
 * @brief Moving the cursor before the start of the source should fail; the reference used for moving the cursor is the
 * beginning of the source.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorBeforeStartReferenceStart) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(-1, std::ios::beg), ov::Exception, _);
}

/**
 * @brief Moving the cursor after the end of the source should fail; the reference used for moving the cursor is the
 * beginning of the source.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorAfterEndReferenceStart) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(blob_content.size() + 1, std::ios::beg), ov::Exception, _);
}

/**
 * @brief Moving the cursor before the start of the source should fail; the reference used for moving the cursor is the
 * current cursor position.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorBeforeStartReferenceCurrent) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(-1, std::ios::cur), ov::Exception, _);
}

/**
 * @brief Moving the cursor after the end of the source should fail; the reference used for moving the cursor is the
 * current cursor position.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorAfterEndReferenceCurrent) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(blob_content.size() + 1, std::ios::cur), ov::Exception, _);
}

/**
 * @brief Moving the cursor before the start of the source should fail; the reference used for moving the cursor is the
 * end of the source.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorBeforeStartReferenceEnd) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(-(static_cast<int64_t>(blob_content.size()) + 1), std::ios::end),
                    ov::Exception,
                    _);
}

/**
 * @brief Moving the cursor after the end of the source should fail; the reference used for moving the cursor is the
 * end of the source.
 */
TEST_P(BlobSourceDifferentBlobsCommon, MoveCursorAfterEndReferenceEnd) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.seekg(1, std::ios::end), ov::Exception, _);
}

/**
 * @brief Cannot extract data without copying if the underlying buffer is not contiguous.
 */
TEST_P(BlobSourceDifferentBlobsNonContiguous, ReadViewFails) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.read_view(0), ov::Exception, _);
}

/**
 * @brief Cannot extract RoI tensors without copying if the underlying buffer is not contiguous.
 */
TEST_P(BlobSourceDifferentBlobsNonContiguous, GetROITensorFails) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.create_roi_tensor(0), ov::Exception, _);
}

/**
 * @brief If the underlying buffer is not contiguous, then the "is_contiguous" call should return "false"
 */
TEST_P(BlobSourceDifferentBlobsNonContiguous, FalseIsContiguous) {
    BlobSource blob_source = create_blob_source();
    ASSERT_FALSE(blob_source.is_contiguous());
}

/**
 * @brief Extract the first byte without copying
 */
TEST_P(BlobSourceDifferentBlobsContiguous, ReadViewFirstByte) {
    BlobSource blob_source = create_blob_source();

    const size_t read_size = 1;
    const char* payload_ptr = nullptr;
    OV_ASSERT_NO_THROW(payload_ptr = reinterpret_cast<const char*>(blob_source.read_view(read_size)));

    std::vector<uint8_t> extracted_payload(payload_ptr, payload_ptr + read_size);
    ASSERT_EQ(extracted_payload, std::vector<uint8_t>(blob_content.begin(), blob_content.begin() + read_size));

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, read_size);
}

/**
 * @brief Extract all bytes without copying
 */
TEST_P(BlobSourceDifferentBlobsContiguous, ReadViewAllBytes) {
    BlobSource blob_source = create_blob_source();

    const char* payload_ptr = nullptr;
    OV_ASSERT_NO_THROW(payload_ptr = reinterpret_cast<const char*>(blob_source.read_view(blob_content.size())));

    std::vector<uint8_t> extracted_payload(payload_ptr, payload_ptr + blob_content.size());
    ASSERT_EQ(extracted_payload, blob_content);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size());
}

/**
 * @brief Move the read cursor and then extract one byte without copying
 */
TEST_P(BlobSourceDifferentBlobsContiguous, ReadViewFirstByteAfterMove) {
    BlobSource blob_source = create_blob_source();
    blob_source.seekg(1);

    const size_t read_size = 1;
    const char* payload_ptr = nullptr;
    OV_ASSERT_NO_THROW(payload_ptr = reinterpret_cast<const char*>(blob_source.read_view(read_size)));

    std::vector<uint8_t> extracted_payload(payload_ptr, payload_ptr + read_size);
    ASSERT_EQ(extracted_payload, std::vector<uint8_t>(blob_content.begin() + 1, blob_content.begin() + 1 + read_size));

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, read_size + 1);
}

/**
 * @brief Attempting to extract more bytes than available without copying should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsContiguous, ReadViewTooMuch) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.read_view(blob_content.size() + 1), ov::Exception, _);
}

/**
 * @brief Attempting to extract data (without copying) beyond the end limit should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsContiguous, ReadViewAfterEnd) {
    BlobSource blob_source = create_blob_source();
    blob_source.seekg(0, std::ios::end);
    OV_EXPECT_THROW(blob_source.read_view(1), ov::Exception, _);
}

/**
 * @brief Extract an roi tensor containing only the first byte of the source
 */
TEST_P(BlobSourceDifferentBlobsContiguous, GetROITensorFirstByte) {
    BlobSource blob_source = create_blob_source();

    const size_t tensor_size = 1;
    ov::Tensor roi_tensor;
    OV_ASSERT_NO_THROW(roi_tensor = blob_source.create_roi_tensor(tensor_size));

    ASSERT_EQ(roi_tensor.get_byte_size(), tensor_size);
    ASSERT_EQ(std::memcmp(roi_tensor.data(), blob_content.data(), 1), 0);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, tensor_size);
}

/**
 * @brief Extract an roi tensor containing the whole source
 */
TEST_P(BlobSourceDifferentBlobsContiguous, GetROITensorAllBytes) {
    BlobSource blob_source = create_blob_source();

    ov::Tensor roi_tensor;
    OV_ASSERT_NO_THROW(roi_tensor = blob_source.create_roi_tensor(blob_content.size()));

    ASSERT_EQ(roi_tensor.get_byte_size(), blob_content.size());
    ASSERT_EQ(std::memcmp(roi_tensor.data(), blob_content.data(), blob_content.size()), 0);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, blob_content.size());
}

/**
 * @brief Extract an roi tensor containing only one byte of the source after moving the cursor
 */
TEST_P(BlobSourceDifferentBlobsContiguous, GetROITensorFirstByteAfterMove) {
    BlobSource blob_source = create_blob_source();
    blob_source.seekg(1);

    const size_t tensor_size = 1;
    ov::Tensor roi_tensor;
    OV_ASSERT_NO_THROW(roi_tensor = blob_source.create_roi_tensor(tensor_size));

    ASSERT_EQ(roi_tensor.get_byte_size(), tensor_size);
    ASSERT_EQ(std::memcmp(roi_tensor.data(), blob_content.data() + 1, 1), 0);

    size_t cursor = 0;
    OV_ASSERT_NO_THROW(cursor = blob_source.tellg());
    ASSERT_EQ(cursor, tensor_size + 1);
}

/**
 * @brief Attempting to extract an roi tensor that is bigger than the available bytes should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsContiguous, GetROITensorTooMuch) {
    BlobSource blob_source = create_blob_source();
    OV_EXPECT_THROW(blob_source.create_roi_tensor(blob_content.size() + 1), ov::Exception, _);
}

/**
 * @brief Attempting to extract an roi tensor beyond the end limit of the source should yield an exception
 */
TEST_P(BlobSourceDifferentBlobsContiguous, GetROITensorAfterEnd) {
    BlobSource blob_source = create_blob_source();
    blob_source.seekg(0, std::ios::end);
    OV_EXPECT_THROW(blob_source.create_roi_tensor(1), ov::Exception, _);
}

/**
 * @brief "is_contiguous" should return true if the underlying buffer is contiguous
 */
TEST_P(BlobSourceDifferentBlobsContiguous, TrueIsContiguous) {
    BlobSource blob_source = create_blob_source();
    ASSERT_TRUE(blob_source.is_contiguous());
}

using BlobSourceContiguous = testing::Test;

TEST_F(BlobSourceContiguous, NotUnidimensionalTensorFails) {
    const auto tensor = ov::Tensor(ov::element::Type_t::u8, ov::Shape({2, TEST_BUFFER.size() / 2}), TEST_BUFFER.data());
    OV_EXPECT_THROW(BlobSource blob_source(tensor), ov::Exception, _);
}

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         BlobSourceDifferentBlobsCommon,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::ValuesIn(ALL_BLOB_SOURCE_DATA_TYPES)),
                         BlobSourceDifferentBlobsCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         BlobSourceDifferentBlobsNonContiguous,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::Values(BlobSourceDataType::STREAM)),
                         BlobSourceDifferentBlobsCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(UnitTests,
                         BlobSourceDifferentBlobsContiguous,
                         testing::Combine(testing::ValuesIn(ALL_BLOB_CONTENT_TYPES),
                                          testing::Values(BlobSourceDataType::TENSOR)),
                         BlobSourceDifferentBlobsCommon::getTestCaseName);
