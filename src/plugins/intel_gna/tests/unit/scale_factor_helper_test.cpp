// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_factor_helper.hpp"

#include <gtest/gtest.h>

using ScaleFactorHelperTestData =
    std::tuple<std::vector<std::pair<std::string, float>>,  // inputs name and scale_factor
               std::tuple<uint16_t, uint16_t, bool>,        // header version majro,min and info if custom scale
                                                            // factor should be applied
               bool>;                                       // is test for legacy

class GNAApplyInputScaleFactorsTest : public ::testing::TestWithParam<ScaleFactorHelperTestData> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<ScaleFactorHelperTestData>& obj) {
        std::vector<std::pair<std::string, float>> test_inputs;
        std::tuple<uint16_t, uint16_t, bool> header_version_with_aplicability;
        bool legacy;

        std::tie(test_inputs, header_version_with_aplicability, legacy) = obj.param;
        std::string name("Inputs:[");
        for (const auto& input : test_inputs) {
            name += "[" + input.first + ", " + std::to_string(input.second) + "],";
        }
        if (name.back() == ',') {
            name.pop_back();
        }
        auto major = std::to_string(std::get<0>(header_version_with_aplicability));
        auto minor = std::to_string(std::get<1>(header_version_with_aplicability));
        auto applicable = std::string(std::get<2>(header_version_with_aplicability) ? "scale_factor_applicable"
                                                                                    : "scale_factor_not_applicable");
        name += "],Header[" + major + "." + minor + "]," + applicable;
        auto legacy_str = std::string(legacy ? "legacy" : "");
        if (!legacy_str.empty()) {
            name += "," + legacy_str;
        }
        return name;
    }
    void SetUp() override {
        std::tuple<uint16_t, uint16_t, bool> header_version_with_aplicability;
        std::tie(_test_inputs, header_version_with_aplicability, _legacy_scale_factor) = GetParam();

        _header.version.major = std::get<0>(header_version_with_aplicability);
        _header.version.minor = std::get<1>(header_version_with_aplicability);
        _applicable = std::get<2>(header_version_with_aplicability);

        _gna_inputs.Get().clear();
        for (const auto& input : _test_inputs) {
            _gna_inputs.Get().push_back({input.first});
        }

        if (_legacy_scale_factor) {
            auto& confg_input_factors = _config.inputScaleFactors;
            confg_input_factors.clear();

            std::transform(_test_inputs.begin(),
                           _test_inputs.end(),
                           std::back_inserter(confg_input_factors),
                           [](const std::pair<std::string, float>& input) {
                               return input.second;
                           });

        } else {
            auto& confg_input_factors = _config.inputScaleFactorsPerInput;
            confg_input_factors.clear();

            // cofingure config with proper scale factors for legacy
            std::transform(_test_inputs.begin(),
                           _test_inputs.end(),
                           std::inserter(confg_input_factors, confg_input_factors.end()),
                           [](const std::pair<std::string, float>& input) {
                               return std::make_pair(input.first, input.second);
                           });
        }
    }

    GNAPluginNS::Config _config;
    GNAPluginNS::HeaderLatest::ModelHeader _header;
    GNAPluginNS::GnaInputs _gna_inputs;
    std::vector<std::pair<std::string, float>> _test_inputs;
    bool _applicable;
    bool _legacy_scale_factor;
};

TEST_P(GNAApplyInputScaleFactorsTest, test_import_scale_factor_set_properly) {
    EXPECT_NO_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(_gna_inputs, _config, _header));
    for (const auto& input : _test_inputs) {
        auto& input_ref = _gna_inputs[input.first];
        if (_applicable) {
            EXPECT_FLOAT_EQ(input_ref.scale_factor, input.second);
        } else {
            EXPECT_FLOAT_EQ(input_ref.scale_factor, GNAPluginNS::kScaleFactorDefault);
        }
    }
}

TEST_P(GNAApplyInputScaleFactorsTest, test_load_scale_factor_set_properly) {
    EXPECT_NO_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(_gna_inputs, _config));
    for (const auto& input : _test_inputs) {
        auto& input_ref = _gna_inputs[input.first];
        EXPECT_FLOAT_EQ(input_ref.scale_factor, input.second);
    }
}

TEST_P(GNAApplyInputScaleFactorsTest, inputs_count_not_match_to_scale_factors) {
    if (_legacy_scale_factor) {
        _config.inputScaleFactors.push_back(10.0);
    } else {
        _config.inputScaleFactorsPerInput["unknown"] = 10.0;
    }

    if (_applicable) {
        EXPECT_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(_gna_inputs, _config, _header), std::exception);
    } else {
        // check if no exception and data are not changed
        EXPECT_NO_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(_gna_inputs, _config, _header));
        for (const auto& input : _test_inputs) {
            auto& input_ref = _gna_inputs[input.first];
            EXPECT_FLOAT_EQ(input_ref.scale_factor, GNAPluginNS::kScaleFactorDefault);
        }
    }
}

std::vector<std::vector<std::pair<std::string, float>>> inputs = {{{"input", 2.0}},
                                                                  {{"input_1", 3.0}, {"input_2", 4.0}}};

// custom scale factors should not be applied for Model with version > 2.7
std::vector<std::tuple<uint16_t, uint16_t, bool>> model_versions = {{2, 6, true}, {2, 7, true}, {2, 8, false}};

INSTANTIATE_TEST_CASE_P(smoke_,
                        GNAApplyInputScaleFactorsTest,
                        ::testing::Combine(::testing::ValuesIn(inputs),
                                           ::testing::ValuesIn(model_versions),
                                           ::testing::ValuesIn({true, false})),
                        GNAApplyInputScaleFactorsTest::get_test_case_name);

// Tests if nothing was changed if there is no custom scale factor in configuration
TEST(smoke_GNAApplyInputScaleFactorsTest, test_default_scale_factor) {
    GNAPluginNS::Config config;
    GNAPluginNS::HeaderLatest::ModelHeader header;
    GNAPluginNS::GnaInputs inputs;

    std::string input_name = "input";
    auto& input_ref = inputs[input_name];
    config.inputScaleFactors.clear();
    config.inputScaleFactorsPerInput.clear();

    EXPECT_NO_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(inputs, config, header));

    EXPECT_FLOAT_EQ(input_ref.scale_factor, GNAPluginNS::kScaleFactorDefault);
}

// Tests if exception is thron if input scale factor name does not match to input name
TEST(smoke_GNAApplyInputScaleFactorsTest, test_wrong_scale_factor_config_input_name) {
    GNAPluginNS::Config config;
    GNAPluginNS::HeaderLatest::ModelHeader header;
    header.version.major = 2;
    header.version.minor = 7;
    GNAPluginNS::GnaInputs gna_inputs;

    const std::string name_1{"input_1"};
    const std::string name_2{"input_2"};
    gna_inputs.Get().clear();
    gna_inputs.Get().push_back(name_1);
    gna_inputs.Get().push_back(name_2);
    config.inputScaleFactorsPerInput.clear();
    config.inputScaleFactorsPerInput[name_1] = 2.0;
    config.inputScaleFactorsPerInput[name_2 + "__"] = 2.0;

    EXPECT_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(gna_inputs, config, header), std::exception);
}

// Tests if input scale factor name matches to the tensor name
TEST_P(GNAApplyInputScaleFactorsTest, test_scale_factor_config_input_tensor_name) {
    GNAPluginNS::Config config;
    GNAPluginNS::GnaInputs gna_inputs;

    const std::string name_1{"input_1"};
    const std::string tensor_name_1{"input_1:0"};

    const std::string name_2{"input_2"};
    const std::string tensor_name_2{"input_2:0"};

    gna_inputs[name_1].tensor_names = {tensor_name_1};
    gna_inputs[name_2].tensor_names = {tensor_name_2};

    config.inputScaleFactorsPerInput.clear();
    config.inputScaleFactorsPerInput[tensor_name_1] = 2.0;
    config.inputScaleFactorsPerInput[name_2] = 2.0;

    EXPECT_NO_THROW(ov::intela_gna::helpers::ApplyInputScaleFactors(gna_inputs, config));

    for (const auto& input : gna_inputs.Get()) {
        EXPECT_FLOAT_EQ(input.scale_factor, 2.0);
    }
}