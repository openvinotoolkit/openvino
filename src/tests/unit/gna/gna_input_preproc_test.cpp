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
    std::shared_ptr<GNAPluginNS::GnaInputs> getInputsPtr() {
        return this->inputs_ptr_;
    }
    void setLowPrc() {
        this->gnaFlags->input_low_precision = true;
    }
};

template<typename T, typename U>
class GNAInputPrecisionTest: public ::testing::Test {
public:
    virtual void SetUp(std::vector<U>& input_vals, std::vector<T>& ref_vals, std::map<std::string, std::string>& config) {
        inputVals = input_vals;
        referenceVals = ref_vals;
        gna_config = config;
        shape = {1, input_vals.size()};
        plugin = GNAPluginForInPrecisionTest(gna_config);
    }

    void Run() {
        InferenceEngine::BlobMap input;
        input["Parameter"] = GenerateInputFromVec(inputVals);

        auto is1D = input.begin()->second->getTensorDesc().getLayout() == Layout::C;
        auto is3D = input.begin()->second->getTensorDesc().getLayout() == Layout::CHW;
        auto dims = input.begin()->second->getTensorDesc().getDims();
        auto  importedElements = is1D ? dims[0] : details::product(++std::begin(dims), std::end(dims));

        auto sf = std::stod(gna_config["GNA_SCALE_FACTOR_0"]);

        auto  importedFrames = (is3D || is1D) ? 1 : dims[0];
        auto  targetGroups = is1D ? 1 : dims[0];

        T *plugin_inputs = new T[shape[1]];

        plugin.ImportFrames(plugin_inputs,
                            input.begin()->second->cbuffer().as<float *>(),
                            input.begin()->second->getTensorDesc().getPrecision(),
                            sf,
                            orientation,
                            importedFrames,
                            targetGroups,
                            importedElements,
                            importedElements);

        for (int i = 0; i < shape[1]; ++i) {
            if (plugin_inputs[i] != referenceVals[i]) {
                std::ostringstream err;
                err << "Actual and reference value of input doesn't match: " << plugin_inputs[i] << " vs "
                    << referenceVals[i];
                delete [] plugin_inputs;
                FAIL() << err.str();
            }
        }
        delete [] plugin_inputs;
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInputFromVec(std::vector<U>& values) const {
        Precision prc;
        if (std::is_same<U, float>::value) {
            prc = Precision::FP32;
        } else if (std::is_same<U, uint16_t>::value) {
            prc = Precision::I16;
        } else {
            prc = Precision::U8;
        }
        TensorDesc desc = {prc, shape, Layout::ANY};
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(desc);
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<U*>();
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

    ngraph::element::Type netPrecision = ngraph::element::f32;
    std::shared_ptr<ov::Model> function;
    CNNNetwork cnnNet;
    SizeVector shape = {1, 8};
    std::map<std::string, std::string> gna_config;
    std::vector<T> referenceVals;
    std::vector<U> inputVals;
    intel_dnn_orientation_t orientation = kDnnInterleavedOrientation;
    GNAPluginForInPrecisionTest plugin;
};

using GNAInputPrecisionTestFp32toI16 = GNAInputPrecisionTest<uint16_t, float>;
using GNAInputPrecisionTestFp32toI8 = GNAInputPrecisionTest<uint8_t, float>;
using GNAInputPrecisionTestFp32toFp32 = GNAInputPrecisionTest<float, float>;

TEST_F(GNAInputPrecisionTestFp32toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "8"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<float> inputs(8, 16);
    std::vector<uint16_t> refs(8, 128);
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
    std::vector<uint8_t> refs(8, 48);
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

using GNAInputPrecisionTestI16toI16 = GNAInputPrecisionTest<uint16_t, uint16_t>;
using GNAInputPrecisionTestI16toI8 = GNAInputPrecisionTest<uint8_t, uint16_t>;

TEST_F(GNAInputPrecisionTestI16toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<uint16_t> inputs(8, 16);
    std::vector<uint16_t> refs(8, 16);
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
    std::vector<uint16_t> inputs(8, 12);
    std::vector<uint8_t> refs(8, 120);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    plugin.setLowPrc();
    Run();
}

using GNAInputPrecisionTestI8toI16 = GNAInputPrecisionTest<uint16_t, uint8_t>;
using GNAInputPrecisionTestI8toI8 = GNAInputPrecisionTest<uint8_t, uint8_t>;

TEST_F(GNAInputPrecisionTestI8toI16, GNAInputPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "8"},
        {"GNA_PRECISION", "I16"}
    };
    std::vector<uint8_t> inputs(8, 16);
    std::vector<uint16_t> refs(8, 128);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    Run();
}

TEST_F(GNAInputPrecisionTestI8toI8, GNAInputPrecisionTestI8) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I8"}
    };
    std::vector<uint8_t> inputs(8, 120);
    std::vector<uint8_t> refs(8, 120);
    SetUp(inputs, refs, config);
    plugin.InitGNADevice();
    plugin.setLowPrc();
    Run();
}
} // namespace testing