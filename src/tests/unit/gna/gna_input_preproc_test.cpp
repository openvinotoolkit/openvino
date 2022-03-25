// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <vector>
#include <map>

#include "gna_plugin.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForInPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
using GNAPlugin::GNAPlugin;
using GNAPlugin::ImportFrames;
    void setLowPrc() {
        this->gnaFlags->input_low_precision = true;
    }
    void setGNADeviceHelper(){
        this->gnadevice = std::make_shared<GNADeviceHelper>();
    }
};

template<typename U, typename T>
class GNAInputPrecisionTest: public ::testing::Test {
public:
    void Compare(GNAPluginForInPrecisionTest *gna_plugin) {
        auto sf = std::stof(gna_config["GNA_SCALE_FACTOR_0"]);
        std::vector<T> plugin_inputs(shape[1]);
        gna_plugin->ImportFrames(&(plugin_inputs.front()),
                            &(inputVals.front()),
                            prc,
                            sf,
                            orientation,
                            shape[0],
                            shape[0],
                            shape[1],
                            shape[1]);

        for (int i = 0; i < shape[1]; ++i) {
            EXPECT_EQ(plugin_inputs[i], referenceVals[i]);
        }
    }

protected:
    InferenceEngine::Precision prc;
    InferenceEngine::SizeVector shape = {1, 8};
    std::map<std::string, std::string> gna_config;
    std::vector<T> referenceVals;
    std::vector<U> inputVals;
    intel_dnn_orientation_t orientation = kDnnInterleavedOrientation;
};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<float, int16_t>;
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<float, int8_t>;
using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_F(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    gna_config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "8"},
        {"GNA_PRECISION", "I16"}
    };
    inputVals = std::vector<float>(8, 16);
    referenceVals = std::vector<int16_t>(8, 128);
    prc = InferenceEngine::Precision::FP32;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    plugin.setGNADeviceHelper();
    Compare(&plugin);
}

TEST_F(GNAInputPrecisionTestFp32toI8, GNAInputPrecisionTestI8) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_SCALE_FACTOR_0", "4"},
                   {"GNA_PRECISION", "I8"}
    };
    inputVals =  std::vector<float>(8, 12);
    referenceVals = std::vector<int8_t>(8, 48);
    prc = InferenceEngine::Precision::FP32;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    plugin.setGNADeviceHelper();
    plugin.setLowPrc();
    Compare(&plugin);
}

TEST_F(GNAInputPrecisionTestFp32toFp32, GNAInputPrecisionTestFp32) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
                   {"GNA_SCALE_FACTOR_0", "1"}
    };
    inputVals = std::vector<float>(8, 1200);
    referenceVals = std::vector<float>(8, 1200);
    prc = InferenceEngine::Precision::FP32;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    Compare(&plugin);
}

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<int16_t, int16_t>;
using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<int16_t, int8_t>;

TEST_F(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_SCALE_FACTOR_0", "1"},
                   {"GNA_PRECISION", "I16"}
    };
    inputVals = std::vector<int16_t>(8, 16);
    referenceVals = std::vector<int16_t>(8, 16);
    prc = InferenceEngine::Precision::I16;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    Compare(&plugin);
}

TEST_F(GNAInputPrecisionTestI16toI8, GNAInputPrecisionTestI8) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_SCALE_FACTOR_0", "10"},
                   {"GNA_PRECISION", "I8"}
    };
    inputVals = std::vector<int16_t>(8, 12);
    referenceVals = std::vector<int8_t>(8, 120);
    prc = InferenceEngine::Precision::I16;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    plugin.setLowPrc();
    Compare(&plugin);
}

using GNAInputPrecisionTestU8toI16 = GNAInputPrecisionTest<uint8_t, int16_t>;
using GNAInputPrecisionTestU8toI8 = GNAInputPrecisionTest<uint8_t, int8_t>;

TEST_F(GNAInputPrecisionTestU8toI16, GNAInputPrecisionTestI16) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_SCALE_FACTOR_0", "8"},
                   {"GNA_PRECISION", "I16"}
    };
    inputVals = std::vector<uint8_t>(8, 16);
    referenceVals = std::vector<int16_t>(8, 128);
    prc = InferenceEngine::Precision::U8;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    Compare(&plugin);
}

TEST_F(GNAInputPrecisionTestU8toI8, GNAInputPrecisionTestI8) {
    gna_config = { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                   {"GNA_SCALE_FACTOR_0", "1"},
                   {"GNA_PRECISION", "I8"}
    };
    inputVals = std::vector<uint8_t>(8, 120);
    referenceVals = std::vector<int8_t>(8, 120);
    prc = InferenceEngine::Precision::U8;
    auto plugin = GNAPluginForInPrecisionTest(gna_config);
    plugin.setLowPrc();
    Compare(&plugin);
}
} // namespace testing