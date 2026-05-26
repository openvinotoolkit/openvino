// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tlv_format.hpp"

#include <gtest/gtest.h>

#include <array>
#include <sstream>

namespace ov::test {
namespace {
std::stringstream make_stream() {
    return std::stringstream(std::ios::in | std::ios::out | std::ios::binary);
}
}  // namespace

using runtime::TLVTraits;

TEST(TLVFormatTest, WriteReadRecordWithRawData) {
    auto stream = make_stream();
    const TLVTraits::TagType expected_tag = 77;
    const std::vector<char> expected_data = {0x12, 0x34, 0x56, 0x78};

    runtime::write_tlv_record(stream, expected_tag, expected_data.size(), expected_data.data());
    stream.seekg(0);

    TLVTraits::TagType tag = 0;
    TLVTraits::LengthType size = 0;
    std::vector<char> data;
    ASSERT_TRUE(runtime::read_tlv_record(stream, tag, size, data));
    EXPECT_EQ(tag, expected_tag);
    EXPECT_EQ(size, expected_data.size());
    EXPECT_EQ(data, expected_data);
    EXPECT_EQ(stream.tellg(), sizeof(TLVTraits::TagType) + sizeof(TLVTraits::LengthType) + expected_data.size());
}

TEST(TLVFormatTest, WriteReadRecordWithWriter) {
    auto stream = make_stream();
    const TLVTraits::TagType expected_tag = 101;
    const std::string expected_data = "payload_data";

    runtime::write_tlv_record(stream, expected_tag, [&](std::ostream& s) {
        s.write(expected_data.data(), expected_data.size());
    });
    stream.seekg(0);

    TLVTraits::TagType tag = 0;
    TLVTraits::LengthType size = 0;
    std::vector<char> buffer;
    ASSERT_TRUE(runtime::read_tlv_record(stream, tag, size, buffer));
    const std::string data{buffer.begin(), buffer.end()};
    EXPECT_EQ(tag, expected_tag);
    EXPECT_EQ(size, expected_data.size());
    EXPECT_EQ(data, expected_data);
    EXPECT_EQ(stream.tellg(), sizeof(TLVTraits::TagType) + sizeof(TLVTraits::LengthType) + expected_data.size());
}

TEST(TLVFormatTest, WriteReadRecordWithZeroSize) {
    auto stream = make_stream();
    const TLVTraits::TagType expected_tag = 15;
    const auto validate = [&]() {
        stream.seekg(0);
        TLVTraits::TagType tag = 0;
        TLVTraits::LengthType size = 123;
        std::vector<char> data = {'A', 'B', 'C'};
        ASSERT_TRUE(runtime::read_tlv_record(stream, tag, size, data));
        EXPECT_EQ(tag, expected_tag);
        EXPECT_EQ(size, 0);
        EXPECT_TRUE(data.empty());
        stream.clear();
    };

    runtime::write_tlv_record(stream, expected_tag, 0, nullptr);
    validate();

    runtime::write_tlv_record(stream, expected_tag, [](std::ostream&) {});
    validate();

    runtime::write_tlv_record(stream, expected_tag, {});
    validate();
}

TEST(TLVFormatTest, ReadRecordReturnsFalseForTruncatedHeader) {
    auto stream = make_stream();
    const TLVTraits::TagType tag = 11;
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.seekg(0);

    TLVTraits::TagType read_tag = 0;
    TLVTraits::LengthType read_size = 0;
    std::vector<char> data;
    EXPECT_FALSE(runtime::read_tlv_record(stream, read_tag, read_size, data));
}

TEST(TLVFormatTest, ReadRecordReturnsFalseForTruncatedPayload) {
    auto stream = make_stream();
    const TLVTraits::TagType tag = 17;
    const TLVTraits::LengthType size = 6;
    const std::array<uint8_t, 3> partial_data = {0xAA, 0xBB, 0xCC};

    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(reinterpret_cast<const char*>(partial_data.data()), partial_data.size());
    stream.seekg(0);

    TLVTraits::TagType read_tag = 0;
    TLVTraits::LengthType read_size = 0;
    std::vector<char> data;
    EXPECT_FALSE(runtime::read_tlv_record(stream, read_tag, read_size, data));
}

TEST(TLVFormatTest, ScanRecordsReadsMatchingTagsAndSkipsUnknown) {
    auto stream = make_stream();
    const std::string first_data = "one";
    const std::string second_data = "skip_me";
    const std::string third_data = "three";

    runtime::write_tlv_record(stream, 1, first_data.size(), first_data.data());
    runtime::write_tlv_record(stream, 2, second_data.size(), second_data.data());
    runtime::write_tlv_record(stream, 3, third_data.size(), third_data.data());

    size_t scanner_calls = 0;
    std::string scanned_first;
    std::string scanned_third;
    runtime::TLVValueScanner scanners;
    scanners.emplace(1, [&](std::istream& s, TLVTraits::LengthType size) {
        scanned_first.resize(size);
        s.read(scanned_first.data(), size);
        ++scanner_calls;
        return s.good();
    });
    scanners.emplace(3, [&](std::istream& s, TLVTraits::LengthType size) {
        scanned_third.resize(size);
        s.read(scanned_third.data(), size);
        ++scanner_calls;
        return s.good();
    });

    stream.seekg(0);
    runtime::scan_tlv_records(stream, scanners);
    EXPECT_EQ(scanner_calls, 2);
    EXPECT_EQ(scanned_first, first_data);
    EXPECT_EQ(scanned_third, third_data);
}
}  // namespace ov::test
