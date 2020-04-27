// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

#include "precision_utils.h"
#include "common_test_utils/xml_net_builder/xml_net_builder.hpp"

using namespace std;
using namespace CommonTestUtils;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

static constexpr int testDimW = 17;
static constexpr int testDimH = 3;

struct in_params {
    std::vector<size_t> in;
};

struct base_layout_test_params {
    std::string device;
    Layout layout;
    std::string precision;
    base_layout_test_params(std::string name, std::string _precision = "FP32", Layout _layout = Layout::C) {
        device = name;
        precision = _precision;
        layout = _layout;
    }
};

struct power_params : in_params {
    int power;
    int scale;
    int shift;

    power_params(in_params in,
        int _power,
        int _scale,
        int _shift ) :
        in_params(in) {
        power = _power;
        scale = _scale;
        shift = _shift;
    }
};

struct layout_test_params : power_params, base_layout_test_params {
    layout_test_params(std::string name, std::string precision, Layout layout, power_params params) :
        power_params(params), base_layout_test_params(name, precision, layout) {}
};

std::ostream &operator<<(std::ostream &os, const layout_test_params &p) {
    return os << "device: " << p.device;
}

std::string layoutToString(Layout l) {
    std::string str_layout;
    switch (l)
    {
    case InferenceEngine::NCHW:
        str_layout = "NCHW";
        break;
    case InferenceEngine::NHWC:
        str_layout = "NHWC";
        break;
    case InferenceEngine::C:
        str_layout = "C";
        break;
    case InferenceEngine::CHW:
        str_layout = "CHW";
        break;
    case InferenceEngine::HW:
        str_layout = "HW";
        break;
    case InferenceEngine::NC:
        str_layout = "NC";
        break;
    case InferenceEngine::CN:
        str_layout = "CN";
        break;
    default:
        break;
    }
    return str_layout;
}

std::string getTestName(testing::TestParamInfo<layout_test_params> obj) {
    return  "layout_" + layoutToString(obj.param.layout) + "_" + obj.param.device;
}

class LayoutTestCanLoad : public TestsCommon,
    public testing::WithParamInterface<layout_test_params>{
protected:
    std::string getPowerModel(const layout_test_params &p) {
        std::map<std::string, std::string> params = {
            { "power", std::to_string(p.power) },
            { "scale", std::to_string(p.scale) },
            { "shift", std::to_string(p.shift) }
        };

        InOutShapes inout = {{p.in},
                           {p.in}};

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Power_Only", inout.inDims[0], p.precision)
            .addLayer("Power", p.precision, &params, inout);
        return model.finish(false);
    }

    std::string getConvModel(const layout_test_params &p) {
        std::map<std::string, std::string> params = {
            { "stride-x", "1" },
            { "stride-y", "1" },
            { "pad-x",    "0" },
            { "pad-y",    "0" },
            { "kernel-x", "1" },
            { "kernel-y", "1" },
            { "output",   std::to_string(testDimW)},
            { "group",    "1" }
        };

        std::vector<size_t> out = p.in;
        if (out.size() == 1 || out.size() == 3) {
            out[0] = testDimW;
        } else {
            out[1] = testDimW;
        }

        InOutShapes inout = {{p.in},
                           {out}};

        const auto elemSize = p.precision == "FP16" ? sizeof(ie_fp16) : sizeof(float);

        size_t weights = testDimW * testDimH * elemSize;
        size_t biases = testDimW * elemSize;

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Convolution_Only", inout.inDims[0], p.precision)
            .addLayer("Convolution", p.precision, &params, inout, weights, biases);
        return model.finish(false);
    }

    std::string getActivModel(const layout_test_params &p) {
        std::map<std::string, std::string> params = {
            { "type", "sigmoid" }
        };

        InOutShapes inout = {{p.in},
                           {p.in}};

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Activation_Only", inout.inDims[0], p.precision)
            .addLayer("Activation", p.precision, &params, inout);
        return model.finish(false);
    }

    Blob::Ptr getNetworkWeights(const layout_test_params &p) {
        const auto elemSize = p.precision == "FP16" ? sizeof(ie_fp16) : sizeof(float);

        TensorDesc tdesc (Precision::U8, { (testDimW * testDimH + testDimW ) * elemSize }, C);
        TBlob<uint8_t> *weights = new TBlob<uint8_t>(tdesc);
        weights->allocate();
        fill_data(weights->buffer().as<float*>(),
            weights->size() / sizeof(float));
        TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);
        return weights_ptr;
    }
};

    class LayoutTestCanLoadPower : public LayoutTestCanLoad {};
    class LayoutTestCanLoadConv : public LayoutTestCanLoad {};
    class LayoutTestCanLoadActiv : public LayoutTestCanLoad {};

    class LayoutTestCanNotLoadPower : public LayoutTestCanLoad {};
    class LayoutTestCanNotLoadConv : public LayoutTestCanLoad {};


TEST_P(LayoutTestCanLoadPower, NetWithLayout) {
    auto param = GetParam();
    InferenceEngine::Core core;
    std::string model = getPowerModel(param);
    Blob::CPtr weights;
    auto network = core.ReadNetwork(model, weights);

    ASSERT_NO_THROW(ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {}));
}

TEST_P(LayoutTestCanLoadConv, NetWithLayout) {
    auto param = GetParam();
    InferenceEngine::Core core;
    std::string model = getConvModel(param);
    Blob::Ptr weights = getNetworkWeights(param);
    auto network = core.ReadNetwork(model, weights);
    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        std::cout << "Device" << param.device << " threw exception \"" << ex.what() << "\" with status code " << ex.getStatus() << std::endl;
        GTEST_FAIL() << ex.what();
    } catch (std::exception ex) {
        std::cout << "Caught" << ex.what() << std::endl;
        GTEST_FAIL() << ex.what();
    } catch (...) {
        GTEST_FAIL();
    }
}


TEST_P(LayoutTestCanLoadActiv, NetWithLayout) {
    auto param = GetParam();
    InferenceEngine::Core core;
    std::string model = getActivModel(param);
    CNNNetwork network;
    Blob::CPtr weights;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, weights));
    ASSERT_NO_THROW(ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {}));
}


TEST_P(LayoutTestCanNotLoadPower, NetWithLayout) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_MYRIAD) ? "Unsupported 1D dimensions" :
                       (param.device == CommonTestUtils::DEVICE_FPGA) ? "Graph is not supported on FPGA plugin due to existance of layer (Name: Input0, Type: Input)\n"\
                            "in topology. Most likely you need to use heterogeneous plugin instead of FPGA plugin directly." : "Invalid data dimensions";
    InferenceEngine::Core core;
    std::string model = getPowerModel(param);
    CNNNetwork network;
    Blob::CPtr weights;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, weights));

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        std::cout << "Device" << param.device << " threw exception \"" << ex.what() << "\" with status code " << ex.getStatus() << std::endl;
        //ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    } catch (std::exception ex) {
        std::cout << "Caught" << ex.what() << std::endl;
        GTEST_FAIL() << ex.what();
    } catch (...) {
        GTEST_FAIL();
    }
}

TEST_P(LayoutTestCanNotLoadConv, NetWithLayout) {
    auto param = GetParam();
    string ref_error =
        (param.device == CommonTestUtils::DEVICE_MYRIAD) ? "Convolution supports only 3D or 4D or 5D input" :
        (param.device == CommonTestUtils::DEVICE_FPGA) ? "Graph is not supported on FPGA" :
        (param.device == CommonTestUtils::DEVICE_CPU) ? "Convolution layer. Unsupported mode. Only 4D and 5D blobs are supported as input." :
        "Invalid data dimensions";
    InferenceEngine::Core core;
    std::string model = getConvModel(param);
    CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, getNetworkWeights(param)));
    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        std::cout << "Device" << param.device << " threw exception \"" << ex.what() << "\" with status code " << ex.getStatus() << std::endl;
        /*if (param.device != CommonTestUtils::DEVICE_CPU) {
            ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        }*/
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    } catch (std::exception ex) {
        std::cout << "Caught" << ex.what() << std::endl;
        GTEST_FAIL() << ex.what();
    } catch (...) {
        GTEST_FAIL();
    }
}
