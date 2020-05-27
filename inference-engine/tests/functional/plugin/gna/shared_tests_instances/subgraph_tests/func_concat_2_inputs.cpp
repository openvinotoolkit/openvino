// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/func_concat_2_inputs.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace LayerTestsDefinitions;
static const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

static std::string serialize_float(float value) {
    std::stringstream val_stream;
    val_stream.imbue(std::locale("C"));
    val_stream << value;
    return val_stream.str();
}

using configMaker_t = std::function<std::map<std::string, std::string>(float)>;

static auto gnaDefaultConfig = [](float scaleFactor) {
    const std::map<std::string, std::string> gnaDefaultConfig = {
        {std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_0", serialize_float(scaleFactor)},
        {std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_1", serialize_float(scaleFactor)},
        {std::string(GNA_CONFIG_KEY(COMPACT_MODE)),        "NO"},
        {GNA_CONFIG_KEY(DEVICE_MODE), GNA_CONFIG_VALUE(SW_EXACT)}
    };

    return gnaDefaultConfig;
};

static auto gnaSwConfig = [](float scaleFactor) {
    static const std::map<std::string, std::string> gnaSwConfig = {
        {GNA_CONFIG_KEY(DEVICE_MODE), GNA_CONFIG_VALUE(SW_FP32)},
        {std::string(GNA_CONFIG_KEY(COMPACT_MODE)),        "NO"},
    };
    return gnaSwConfig;
};

std::vector<std::vector<size_t>> CombineVectors(std::vector<size_t> && a, std::vector<size_t> && b) {
    std::vector<std::vector<size_t>> cartesian;
    std::for_each(a.begin(), a.end(),
                  [&cartesian, &b](size_t & value1) {
                      std::for_each(b.begin(), b.end(),
                                    [&cartesian, &value1](size_t& value2) {
                                        cartesian.emplace_back(std::vector<size_t>{value1, value2});
                      });
                  });
    return cartesian;
}

auto usingConfig = [] (std::string name, float threshold, float range, const configMaker_t configMaker) {
    return ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(CombineVectors(
            // Size of first concat input
            {64, 32, 31, 49},
            // Size of second concat input
            {64, 32, 31, 73, 65})),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(name),
        ::testing::Values(threshold),
        ::testing::Values(range),
        ::testing::Values(configMaker(16384.f / range)));
};

INSTANTIATE_TEST_CASE_P(
    NonTrivialConcat2InputsDefault,
    NonTrivialConcat2Inputs,
    usingConfig("defaultConfig", 0.5f, 10, gnaDefaultConfig),
    NonTrivialConcat2Inputs::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    NonTrivialConcat2InputsFp32,
    NonTrivialConcat2Inputs,
    usingConfig("sf_fp32Config", 1e-3 , 10, gnaSwConfig),
    NonTrivialConcat2Inputs::getTestCaseName);