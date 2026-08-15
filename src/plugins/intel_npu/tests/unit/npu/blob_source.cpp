// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_source.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

using namespace intel_npu;

namespace {

enum class BlobContentType { STANDARD_STRING, SPECIAL_CHARS_STRING, BUFFER };

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
            return;
        }
        case BlobContentType::SPECIAL_CHARS_STRING: {
            blob_content = TEST_STRING_SPECIAL_CHARS;
            return;
        }
        case BlobContentType::BUFFER: {
            blob_content = TEST_BUFFER;
            return;
        }
        default: {
            OPENVINO_THROW(INVALID_BLOB_TYPE_MESSAGE);
        }
        }
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
};

struct BlobFormatImportersTest : public ::testing::Test {
    BlobFormatImportersTest() : config(std::make_shared<OptionsDesc>()) {}

    std::unique_ptr<IBlobFormatImporter> importer;
    FilteredConfig config;
};

/**
 * @brief Empty blobs should not be accepted by the importer factory
 */
TEST_F(BlobFormatImportersTest, FactoryEmptyInputFails) {
    const std::string empty_buffer("");
    std::istringstream input_stream(empty_buffer);
    BlobSource source(input_stream);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({0}), empty_buffer.data());
    source = BlobSource(input_tensor);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);
}
