// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "gna_plugin.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForInPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
using GNAPlugin::GNAPlugin;
using GNAPlugin::ImportFrames;
using GNAPlugin::InitGNADevice;
    void setLowPrc() {
        this->gnaFlags->input_low_precision = true;
    }
};

template<typename U, typename T>
class GNAInputPrecisionTest: public ::testing::Test {
public:
    virtual void SetUp(std::vector<U>& input_vals, std::vector<T>& ref_vals, std::map<std::string, std::string>& config) {
        inputVals = input_vals;
        referenceVals = ref_vals;
        gna_config = config;
        shape = {1, input_vals.size()};
        plugin = GNAPluginForInPrecisionTest(gna_config);
        if (std::is_same<U, float>::value) {
            prc = Precision::FP32;
        } else if (std::is_same<U, int16_t>::value) {
            prc = Precision::I16;
        } else {
            prc = Precision::U8;
        }
    }

    void Run() {
        auto sf = std::stod(gna_config["GNA_SCALE_FACTOR_0"]);
        std::vector<T> plugin_inputs(shape[1]);
        plugin.ImportFrames(&(plugin_inputs.front()),
                            &(inputVals.front()),
                            prc,
                            sf,
                            orientation,
                            shape[0],
                            shape[0],
                            shape[1],
                            shape[1]);

        for (int i = 0; i < shape[1]; ++i) {
            if (plugin_inputs[i] != referenceVals[i]) {
                std::ostringstream err;
                err << "Actual and reference value of input doesn't match: " << plugin_inputs[i] << " vs "
                    << referenceVals[i];
                FAIL() << err.str();
            }
        }
    }

protected:
    Precision prc;
    std::shared_ptr<ov::Model> function;
    CNNNetwork cnnNet;
    SizeVector shape = {1, 8};
    std::map<std::string, std::string> gna_config;
    std::vector<T> referenceVals;
    std::vector<U> inputVals;
    intel_dnn_orientation_t orientation = kDnnInterleavedOrientation;
    GNAPluginForInPrecisionTest plugin;
};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<float, int16_t>;
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<float, int8_t>;
using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_F(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "8"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<float> inputs(8, 16);
    std::vector<int16_t> refs(8, 128);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    Run();
}

TEST_F(GNAInputPrecisionTestFp32toI8, GNAInputPrecisionTestI8) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "4"},
        {"GNA_PRECISION", "I8"}
    };
    std::vector<float> inputs(8, 12);
    std::vector<int8_t> refs(8, 48);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    plugin.setLowPrc();
    Run();
}

TEST_F(GNAInputPrecisionTestFp32toFp32, GNAInputPrecisionTestFp32) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
        {"GNA_SCALE_FACTOR_0", "1"}
    };
    std::vector<float> inputs(8, 1200);
    std::vector<float> refs(8, 1200);
    SetUp(inputs, refs, config);
    Run();
}

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<int16_t, int16_t>;
using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<int16_t, int8_t>;

TEST_F(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<int16_t> inputs(8, 16);
    std::vector<int16_t> refs(8, 16);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    Run();
}

TEST_F(GNAInputPrecisionTestI16toI8, GNAInputPrecisionTestI8) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "10"},
        {"GNA_PRECISION", "I8"}
    };
    std::vector<int16_t> inputs(8, 12);
    std::vector<int8_t> refs(8, 120);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    plugin.setLowPrc();
    Run();
}

using GNAInputPrecisionTestU8toI16 = GNAInputPrecisionTest<uint8_t, int16_t>;
using GNAInputPrecisionTestU8toI8 = GNAInputPrecisionTest<uint8_t, int8_t>;

TEST_F(GNAInputPrecisionTestU8toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "8"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<uint8_t> inputs(8, 16);
    std::vector<int16_t> refs(8, 128);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    Run();
}

TEST_F(GNAInputPrecisionTestU8toI8, GNAInputPrecisionTestI8) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I8"}
    };
    std::vector<uint8_t> inputs(8, 120);
    std::vector<int8_t> refs(8, 120);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    plugin.setLowPrc();
    Run();
}
} // namespace testing