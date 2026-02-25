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

using runtime::TLVFormat;

TEST(TLVFormatTest, WriteReadEntryWithRawData) {
    auto stream = make_stream();
    const TLVFormat::TagType expected_tag = 77;
    const std::array<uint8_t, 4> expected_data = {0x12, 0x34, 0x56, 0x78};

    TLVFormat::write_entry(stream, expected_tag, expected_data.size(), expected_data.data());
    stream.seekg(0);

    TLVFormat::TagType tag = 0;
    TLVFormat::LengthType size = 0;
    std::vector<uint8_t> data;
    ASSERT_TRUE(TLVFormat::read_entry(stream, tag, size, data));
    EXPECT_EQ(tag, expected_tag);
    EXPECT_EQ(size, expected_data.size());
    EXPECT_EQ(data, expected_data);
    EXPECT_EQ(stream.tellg(), sizeof(TLVFormat::TagType) + sizeof(TLVFormat::LengthType) + expected_data.size());
}

TEST(TLVFormatTest, WriteReadEntryWithWriter) {
    auto stream = make_stream();
    const TLVFormat::TagType expected_tag = 101;
    const std::string expected_data = "payload_data";

    TLVFormat::write_entry(stream, expected_tag, [&](std::ostream& s) {
        s.write(expected_data.data(), expected_data.size());
    });
    stream.seekg(0);

    TLVFormat::TagType tag = 0;
    TLVFormat::LengthType size = 0;
    std::string data;
    ASSERT_TRUE(TLVFormat::read_entry(stream, tag, size, data));
    EXPECT_EQ(tag, expected_tag);
    EXPECT_EQ(size, expected_data.size());
    EXPECT_EQ(data, expected_data);
    EXPECT_EQ(stream.tellg(), sizeof(TLVFormat::TagType) + sizeof(TLVFormat::LengthType) + expected_data.size());
}

TEST(TLVFormatTest, WriteReadEntryWithZeroSize) {
    auto stream = make_stream();
    const TLVFormat::TagType expected_tag = 15;
    const auto validate = [&]() {
        stream.seekg(0);
        TLVFormat::TagType tag = 0;
        TLVFormat::LengthType size = 123;
        std::vector<uint8_t> data = {0xAB};
        ASSERT_TRUE(TLVFormat::read_entry(stream, tag, size, data));
        EXPECT_EQ(tag, expected_tag);
        EXPECT_EQ(size, 0);
        EXPECT_TRUE(data.empty());
        stream.clear();
    };

    TLVFormat::write_entry(stream, expected_tag, 0, nullptr);
    validate();

    TLVFormat::write_entry(stream, expected_tag, [](std::ostream&) {});
    validate();

    TLVFormat::write_entry(stream, expected_tag, {});
    validate();
}

TEST(TLVFormatTest, ReadEntryReturnsFalseForTruncatedHeader) {
    auto stream = make_stream();
    const TLVFormat::TagType tag = 11;
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.seekg(0);

    TLVFormat::TagType read_tag = 0;
    TLVFormat::LengthType read_size = 0;
    std::vector<uint8_t> data;
    EXPECT_FALSE(TLVFormat::read_entry(stream, read_tag, read_size, data));
}

TEST(TLVFormatTest, ReadEntryReturnsFalseForTruncatedPayload) {
    auto stream = make_stream();
    const TLVFormat::TagType tag = 17;
    const TLVFormat::LengthType size = 6;
    const std::array<uint8_t, 3> partial_data = {0xAA, 0xBB, 0xCC};

    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.write(reinterpret_cast<const char*>(partial_data.data()), partial_data.size());
    stream.seekg(0);

    TLVFormat::TagType read_tag = 0;
    TLVFormat::LengthType read_size = 0;
    std::vector<uint8_t> data;
    EXPECT_FALSE(TLVFormat::read_entry(stream, read_tag, read_size, data));
}

TEST(TLVFormatTest, ScanEntriesReadsMatchingTagsAndSkipsUnknown) {
    auto stream = make_stream();
    const std::string first_data = "one";
    const std::string second_data = "skip_me";
    const std::string third_data = "three";

    TLVFormat::write_entry(stream, 1, first_data.size(), reinterpret_cast<const uint8_t*>(first_data.data()));
    TLVFormat::write_entry(stream, 2, second_data.size(), reinterpret_cast<const uint8_t*>(second_data.data()));
    TLVFormat::write_entry(stream, 3, third_data.size(), reinterpret_cast<const uint8_t*>(third_data.data()));

    stream.seekg(0);
    const auto begin_pos = stream.tellg();

    size_t scanner_calls = 0;
    std::string scanned_first;
    std::string scanned_third;
    TLVFormat::ValueScanner scanners;
    scanners.emplace(1, [&](std::istream& s, TLVFormat::LengthType size) {
        scanned_first.resize(size);
        s.read(scanned_first.data(), size);
        ++scanner_calls;
    });
    scanners.emplace(3, [&](std::istream& s, TLVFormat::LengthType size) {
        scanned_third.resize(size);
        s.read(scanned_third.data(), size);
        ++scanner_calls;
    });

    TLVFormat::scan_entries(stream, scanners, true);
    EXPECT_EQ(scanner_calls, 2);
    EXPECT_EQ(scanned_first, first_data);
    EXPECT_EQ(scanned_third, third_data);
    EXPECT_EQ(stream.tellg(), begin_pos);
}

TEST(TLVFormatTest, ScanEntriesWithoutRewindLeavesStreamAtEnd) {
    auto stream = make_stream();
    const std::vector<uint8_t> data = {1, 2, 3, 4};
    TLVFormat::write_entry(stream, 9, data.size(), data.data());

    stream.seekg(0);
    stream.seekg(0, std::ios::end);
    const auto end_pos = stream.tellg();
    stream.seekg(0);

    TLVFormat::scan_entries(stream, {}, false);
    EXPECT_EQ(stream.tellg(), end_pos);
}

}  // namespace ov::test
