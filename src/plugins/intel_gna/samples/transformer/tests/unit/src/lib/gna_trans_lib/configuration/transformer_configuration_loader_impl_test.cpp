// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "configuration/transformer_configuration_loader_impl.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

using namespace transformation_sample;

static const std::string kInvalidJSON = "{";
static const std::string kValidJSONWrongContent = "{}";
static const std::string kProperJSON = R"(
{
    "configuration": {
        "GNA_HW_EXECUTION_TARGET": "GNA_3_5",
            "GNA_PWL_MAX_ERROR_PERCENT" : "0.5",
            "INFERENCE_PRECISION_HINT" : "i8"
    },
    "transformations_list" : [
    { "name": "test_1" },
    { "name": "test_2" } ]
})";

static const std::string kProperJSONFilePath = "test_configuration.json";

class TransformerConfigurationLoaderImplFixture : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
};

void TransformerConfigurationLoaderImplFixture::SetUp() {
    std::ofstream file(kProperJSONFilePath);
    file << kProperJSON;
    file.close();
}

void TransformerConfigurationLoaderImplFixture::TearDown() {
    if (std::ifstream(kProperJSONFilePath)) {
        std::remove(kProperJSONFilePath.c_str());
    }
}

TEST_F(TransformerConfigurationLoaderImplFixture, invalid_json) {
    TransformerConfigurationLoaderImpl loader;
    EXPECT_THROW(loader.parse_configuration(std::istringstream(kValidJSONWrongContent)), std::exception);
}

TEST_F(TransformerConfigurationLoaderImplFixture, proper_json) {
    TransformerConfigurationLoaderImpl loader;
    TransformerConfiguration config;

    ASSERT_NO_THROW({ config = loader.parse_configuration(std::istringstream(kProperJSON)); });
    EXPECT_EQ(config.gna_configuration.size(), 3);
    EXPECT_EQ(config.gna_configuration["GNA_HW_EXECUTION_TARGET"], std::string("GNA_3_5"));
    EXPECT_EQ(config.gna_configuration["GNA_PWL_MAX_ERROR_PERCENT"], std::string("0.5"));
    EXPECT_EQ(config.gna_configuration["INFERENCE_PRECISION_HINT"], std::string("i8"));
    EXPECT_EQ(config.transformations_names.size(), 2);
    EXPECT_TRUE(std::count(config.transformations_names.begin(), config.transformations_names.end(), "test_1") == 1);
    EXPECT_TRUE(std::count(config.transformations_names.begin(), config.transformations_names.end(), "test_2") == 1);
}

TEST_F(TransformerConfigurationLoaderImplFixture, file_not_found) {
    TransformerConfigurationLoaderImpl loader;
    EXPECT_THROW(loader.parse_configuration("dummy_file.txt"), std::exception);
}

TEST_F(TransformerConfigurationLoaderImplFixture, proper_json_file) {
    TransformerConfigurationLoaderImpl loader;
    TransformerConfiguration config;
    ASSERT_NO_THROW({ config = loader.parse_configuration(kProperJSONFilePath); });
    EXPECT_EQ(config.gna_configuration.size(), 3);
    EXPECT_EQ(config.gna_configuration["GNA_HW_EXECUTION_TARGET"], std::string("GNA_3_5"));
    EXPECT_EQ(config.gna_configuration["GNA_PWL_MAX_ERROR_PERCENT"], std::string("0.5"));
    EXPECT_EQ(config.gna_configuration["INFERENCE_PRECISION_HINT"], std::string("i8"));
    EXPECT_EQ(config.transformations_names.size(), 2);
    EXPECT_TRUE(std::count(config.transformations_names.begin(), config.transformations_names.end(), "test_1") == 1);
    EXPECT_TRUE(std::count(config.transformations_names.begin(), config.transformations_names.end(), "test_2") == 1);
}
