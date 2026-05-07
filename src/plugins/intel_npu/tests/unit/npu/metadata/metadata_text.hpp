// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <sstream>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"
#include "metadata_wrappers.hpp"

using namespace intel_npu;

using MetadataUnitTests = ::testing::Test;

namespace {

std::string make_text_string(MetadataBase& meta) {
    std::ostringstream stream;
    meta.write_as_text(stream);
    return stream.str();
}

}  // namespace

using MetadataHumanReadableTests = ::testing::Test;

TEST_F(MetadataHumanReadableTests, minimalMetadata) {
    auto meta = Metadata<METADATA_VERSION_2_0>(0, CURRENT_OPENVINO_VERSION);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_FALSE(storedMeta->get_init_sizes().has_value());
    ASSERT_FALSE(storedMeta->get_batch_size().has_value());
    ASSERT_FALSE(storedMeta->get_input_layouts().has_value());
    ASSERT_FALSE(storedMeta->get_output_layouts().has_value());
    ASSERT_FALSE(storedMeta->get_compiler_version().has_value());
}

TEST_F(MetadataHumanReadableTests, initSizes) {
    const std::vector<uint64_t> initSizes{16, 32, 64};
    auto meta = Metadata<METADATA_VERSION_2_1>(0, CURRENT_OPENVINO_VERSION, initSizes);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_TRUE(storedMeta->get_init_sizes().has_value());
}

TEST_F(MetadataHumanReadableTests, emptyInitSizes) {
    auto meta = Metadata<METADATA_VERSION_2_1>(0, CURRENT_OPENVINO_VERSION, std::vector<uint64_t>{});
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_FALSE(storedMeta->get_init_sizes().has_value());
}

TEST_F(MetadataHumanReadableTests, batchSize) {
    const int64_t batchSize = 4;
    auto meta = Metadata<METADATA_VERSION_2_2>(0, CURRENT_OPENVINO_VERSION, std::nullopt, batchSize);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_TRUE(storedMeta->get_batch_size().has_value());
    EXPECT_EQ(storedMeta->get_batch_size().value(), batchSize);
}

TEST_F(MetadataHumanReadableTests, zeroBatchSize) {
    auto meta = Metadata<METADATA_VERSION_2_2>(0, CURRENT_OPENVINO_VERSION, std::nullopt, std::nullopt);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_FALSE(storedMeta->get_batch_size().has_value());
}

TEST_F(MetadataHumanReadableTests, noLayouts) {
    auto meta = Metadata<METADATA_VERSION_2_3>(0, CURRENT_OPENVINO_VERSION, std::nullopt, std::nullopt, std::nullopt);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_FALSE(storedMeta->get_input_layouts().has_value());
    ASSERT_FALSE(storedMeta->get_output_layouts().has_value());
}

TEST_F(MetadataHumanReadableTests, compilerVersion) {
    const uint32_t compilerVersion = 0xCAFECAFE;
    auto meta = Metadata<METADATA_VERSION_2_4>(0,
                                               CURRENT_OPENVINO_VERSION,
                                               std::nullopt,
                                               std::nullopt,
                                               std::nullopt,
                                               std::nullopt,
                                               compilerVersion);
    const auto text = make_text_string(meta);

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_as_text(text));

    ASSERT_FALSE(storedMeta->get_compiler_version().has_value());
}

TEST_P(MetadataTextTest, Format) {
    std::unique_ptr<MetadataBase> meta;
    if (isValid) {
        OV_ASSERT_NO_THROW(meta = ::read_as_text(compatibilityString));
    } else {
        ASSERT_ANY_THROW(meta = ::read_as_text(compatibilityString));
    }
}
