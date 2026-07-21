// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "common_test_utils/test_assertions.hpp"

using namespace intel_npu;

namespace {}  // namespace

using testing::_;

struct BlobFormatImportersTest : public ::testing::Test {
    BlobFormatImportersTest() : config(std::make_shared<OptionsDesc>()) {}

    FilteredConfig config;
};

TEST_F(BlobFormatImportersTest, FactoryEmptyInputFails) {
    std::istringstream input_stream("");
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_stream, false, nullptr, config), ov::Exception, _);
    ov::Tensor input_tensor;
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_tensor, false, nullptr, config), ov::Exception, _);
}
