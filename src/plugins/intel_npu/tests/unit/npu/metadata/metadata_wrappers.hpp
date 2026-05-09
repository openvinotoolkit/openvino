// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "metadata.hpp"

using namespace intel_npu;

struct MetadataTest : Metadata<CURRENT_METADATA_VERSION> {
    MetadataTest(uint64_t blobSize,
                 const std::optional<OpenvinoVersion>& ovVersion,
                 const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt,
                 const std::optional<int64_t>& batchSize = std::nullopt,
                 const std::optional<std::vector<ov::Layout>>& inputLayouts = std::nullopt,
                 const std::optional<std::vector<ov::Layout>>& outputLayouts = std::nullopt,
                 const std::optional<uint32_t>& compilerVersion = std::nullopt,
                 const std::optional<uint64_t>& blobSizeAfterEncryption = std::nullopt,
                 const std::optional<std::string>& compatibilityDescriptor = std::nullopt)
        : Metadata<CURRENT_METADATA_VERSION>(blobSize,
                                             ovVersion,
                                             initSizes,
                                             batchSize,
                                             inputLayouts,
                                             outputLayouts,
                                             compilerVersion,
                                             blobSizeAfterEncryption,
                                             compatibilityDescriptor) {}

    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }
};

struct MetadataVersionTestFixture : Metadata<CURRENT_METADATA_VERSION>, ::testing::TestWithParam<uint32_t> {
public:
    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }

    MetadataVersionTestFixture() : Metadata<CURRENT_METADATA_VERSION>(0, std::nullopt) {}

    MetadataVersionTestFixture(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion)
        : Metadata<CURRENT_METADATA_VERSION>(blobSize, ovVersion) {}

    void TestBody() override {}

    static std::string getTestCaseName(const testing::TestParamInfo<MetadataVersionTestFixture::ParamType>& info) {
        std::ostringstream result;
        result << "majorVersion=" << MetadataBase::get_major(info.param)
               << "_minorVersion=" << MetadataBase::get_minor(info.param);
        return result.str();
    }

    std::stringstream blob;
};

struct MetadataTextTest : Metadata<CURRENT_METADATA_VERSION>, ::testing::TestWithParam<std::tuple<std::string, bool>> {
public:
    MetadataTextTest() : Metadata<CURRENT_METADATA_VERSION>(0, std::nullopt) {}

    void SetUp() override {
        isValid = std::get<1>(GetParam());
        compatibilityString = std::get<0>(GetParam());
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MetadataTextTest::ParamType>& info) {
        return "compatibilityString=\"" + std::get<0>(info.param) +
               "\"_isValid=" + (std::get<1>(info.param) ? "true" : "false");
    }

    std::string compatibilityString;
    bool isValid;
};
