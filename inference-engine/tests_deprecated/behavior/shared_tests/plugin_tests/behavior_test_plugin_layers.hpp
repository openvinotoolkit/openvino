// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include "common_test_utils/xml_net_builder/xml_net_builder.hpp"

using namespace std;
using namespace CommonTestUtils;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

struct in_params_c {
    struct {
        size_t c;
    } in;
};

struct in_params_nc {
    struct {
        size_t n;
        size_t c;
    } in;
};

struct in_params_chw {
    struct {
        size_t c;
        size_t h;
        size_t w;
    } in;
};

struct in_params_nchw {
    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;
};

struct str_params {
    struct {
        size_t w;
        size_t h;
    } str;
};

struct krn_params {
    struct {
        size_t w;
        size_t h;
    } krn;
};

struct pad_params {
    struct {
        size_t w;
        size_t h;
    } pad;
};

struct base_test_params {
    std::string device;
    std::string precision;

    base_test_params(std::string name, std::string _precision = "FP32") {
        device = name;
        precision = _precision;
    }
};

struct conv_params : in_params_chw, krn_params, str_params, pad_params {
    size_t out_c;
    size_t grp_c;

    struct out_struct {
        size_t w;
        size_t h;
    } out;

    conv_params(in_params_chw in,
                krn_params krn,
                str_params str,
                pad_params pad,
                size_t _out_c,
                size_t _grp_c,
                out_struct _out = {}) :
            in_params_chw(in), krn_params(krn), str_params(str), pad_params(pad) {
        out_c = _out_c;
        grp_c = _grp_c;
        out = _out;
    }
};

struct conv_test_params : conv_params, base_test_params {
    conv_test_params(std::string name, conv_params params) :
            conv_params(params), base_test_params(name) {}
};

template<class T>
class LayerTestsCommon : public TestsCommon,
                         public testing::WithParamInterface<T> {
protected:
};

template<class T>
std::string getTestName(testing::TestParamInfo<T> obj) {
    return obj.param.device;
}

IE_SUPPRESS_DEPRECATED_START
class ConvolutionLayerTest : public LayerTestsCommon<conv_test_params> {
protected:
    std::string getModel(const conv_test_params& p) {
        std::map<std::string, std::string> params = {
                {"stride-x", std::to_string(p.str.w)},
                {"stride-y", std::to_string(p.str.h)},
                {"pad-x",    std::to_string(p.pad.w)},
                {"pad-y",    std::to_string(p.pad.h)},
                {"kernel-x", std::to_string(p.krn.w)},
                {"kernel-y", std::to_string(p.krn.h)},
                {"output",   std::to_string(p.out_c)},
                {"group",    std::to_string(p.grp_c)}
        };
        size_t out_h = p.out.h == 0 ?
                    (p.in.h + 2 * p.pad.h - p.krn.h) / p.str.h + 1 : p.out.h;
        size_t out_w = p.out.w == 0 ?
                    (p.in.w + 2 * p.pad.w - p.krn.w) / p.str.w + 1 : p.out.w;
        InOutShapes inout = {{{p.in.c,  p.in.h, p.in.w}},
                           {{p.out_c, out_h,  out_w}}};

        size_t weights = (p.krn.w * p.krn.h * p.out_c * p.in.c / p.grp_c) *
                         sizeof(float);
        size_t biases = p.out_c * sizeof(float);

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Convolution_Only", inout.inDims[0], p.precision)
                .addLayer("Convolution", p.precision, &params, inout, weights, biases);
        return model.finish();
    }

    InferenceEngine::TBlob<uint8_t>::Ptr GetNetworkWeights(const conv_test_params& p) {
        TBlob<uint8_t>* weights = new TBlob<uint8_t>(
                { Precision::U8, {
                        (p.krn.w * p.krn.h * p.out_c * p.in.c / p.grp_c + p.out_c)
                        * sizeof(float)}, Layout::C} );
        weights->allocate();
        fill_data(weights->buffer().as<float*>(),
                  weights->size() / sizeof(float));
        TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

        return weights_ptr;
    }
};

class DeconvolutionLayerTest : public ConvolutionLayerTest {
protected:
    std::string getModel(const conv_test_params& p) {
        std::map<std::string, std::string> params = {
                {"stride-x", std::to_string(p.str.w)},
                {"stride-y", std::to_string(p.str.h)},
                {"pad-x",    std::to_string(p.pad.w)},
                {"pad-y",    std::to_string(p.pad.h)},
                {"kernel-x", std::to_string(p.krn.w)},
                {"kernel-y", std::to_string(p.krn.h)},
                {"output",   std::to_string(p.out_c)},
                {"group",    std::to_string(p.grp_c)}
        };
        size_t out_h = p.out.h == 0 ?
                    (p.in.h + 2 * p.pad.h - p.krn.h) / p.str.h + 1 : p.out.h;
        size_t out_w = p.out.w == 0 ?
                    (p.in.w + 2 * p.pad.w - p.krn.w) / p.str.w + 1 : p.out.w;
        InOutShapes inout = {{{p.in.c,  p.in.h, p.in.w}},
                           {{p.out_c, out_h,  out_w}}};

        size_t weights = (p.krn.w * p.krn.h * p.out_c * p.in.c / p.grp_c) *
                         sizeof(float);

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Deconvolution_Only", inout.inDims[0], p.precision)
                .addLayer("Deconvolution", p.precision, &params, inout, weights);
        return model.finish();
    }
};

TEST_P(ConvolutionLayerTest, CanNotLoadConvLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Convolution1:Convolution";
    
    InferenceEngine::Core core;
    auto network = core.ReadNetwork(getModel(param), GetNetworkWeights(param));

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

TEST_P(DeconvolutionLayerTest, CanNotLoadDeconvLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA)
                       ? "Graph is not supported on FPGA"
                       : (param.device == CommonTestUtils::DEVICE_GNA)
                         ? "[GNAPlugin] in function LoadNetwork: The plugin does not support layer: Deconvolution1:Deconvolution\n"
                         : "Unsupported layer: Deconvolution1:Deconvolution";

    InferenceEngine::Core core;
    auto network = core.ReadNetwork(getModel(param), GetNetworkWeights(param));

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }

}

#define conv_case conv_params({{32, 16, 9}, {1, 1}, {1, 1}, {0, 0}, 17, 1})
#define conv_dw_case conv_params({{32, 16, 9}, {1, 1}, {2, 2}, {0, 0}, 512, 512})

struct pool_params : in_params_nchw, krn_params, str_params, pad_params {
    pool_params(in_params_nchw in, krn_params krn, str_params str, pad_params pad) :
            in_params_nchw(in), krn_params(krn), str_params(str), pad_params(pad) {}
};

struct pool_test_params : pool_params, base_test_params {
    pool_test_params(std::string name, std::string pr, pool_params params) :
            pool_params(params), base_test_params(name, pr) {}
};

class PoolingLayerTest : public LayerTestsCommon<pool_test_params> {
protected:
    std::string getModel(const pool_test_params& p) {
        std::map<std::string, std::string> params = {
                {"stride-x", std::to_string(p.str.w)},
                {"stride-y", std::to_string(p.str.h)},
                {"pad-x",    std::to_string(p.pad.w)},
                {"pad-y",    std::to_string(p.pad.h)},
                {"kernel-x", std::to_string(p.krn.w)},
                {"kernel-y", std::to_string(p.krn.h)},
                {"method",   "MAX"},
                {"round",    "Ceil"}
        };

        size_t out_h = (p.in.h + 2 * p.pad.h - p.krn.h) / p.str.h + 1;
        size_t out_w = (p.in.w + 2 * p.pad.w - p.krn.w) / p.str.w + 1;
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.h, p.in.w}},
                {{p.in.n, p.in.c, out_h,  out_w}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Pooling_Only", inout.inDims[0], p.precision)
                .addLayer("Pooling", p.precision, &params, inout);
        return model.finish();
    }
};

class ROIPoolingLayerTest : public LayerTestsCommon<pool_test_params> {
protected:
    std::string getROIPoolingModel(const pool_test_params& p) {

        size_t out_h = (p.in.h + 2 * p.pad.h - p.krn.h) / p.str.h + 1;
        size_t out_w = (p.in.w + 2 * p.pad.w - p.krn.w) / p.str.w + 1;
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.h, p.in.w}, {p.in.n, p.in.c}},
                {{p.in.n, p.in.c, out_h,  out_w}}
        };
        std::map<std::string, std::string> params = {
                {"pooled_h",      std::to_string(out_h)},
                {"pooled_w",      std::to_string(out_w)},
                {"spatial_scale", "0.062500"}
        };
        return V2NetBuilder::buildNetworkWithOneInput("ROIPooling_Only", inout.inDims[0], p.precision)
                .addInputLayer(p.precision, inout.inDims[1])
                .addLayer("ROIPooling", p.precision, &params, inout)
                .havingEdges().connect(0, 2).connect(1, 2).finish();
    }
};

TEST_P(PoolingLayerTest, CanNotLoadPoolLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Pooling1:Pooling";
    
    InferenceEngine::Core core;
    std::string model = getModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_EQ(ex.what(), ref_error);
    }
}

TEST_P(ROIPoolingLayerTest, CanNotLoadROIPoolLayer) {
    auto param = GetParam();
    string ref_error =
            (param.device == CommonTestUtils::DEVICE_FPGA) ?
                "Graph is not supported on FPGA" :
                "Unsupported layer: ROIPooling2:ROIPooling";
                
    InferenceEngine::Core core;
    std::string model = getROIPoolingModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    if (param.device == CommonTestUtils::DEVICE_CPU ||
        param.device == CommonTestUtils::DEVICE_MYRIAD ||
        param.device == CommonTestUtils::DEVICE_HDDL ||
        param.device == CommonTestUtils::DEVICE_KEEMBAY) {
        ASSERT_NO_THROW(ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {}));
    } else {
        try {
            ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
        } catch (InferenceEngineException ex) {
            if (param.device != CommonTestUtils::DEVICE_HDDL) {
                ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
            }
            ASSERT_STR_CONTAINS(ex.what(), ref_error);
        }
    }
}

#define pool_case pool_params({{1, 1, 16, 16}, {2, 2}, {2, 2}, {0, 0}})

struct activ_test_params : in_params_nchw, base_test_params {
    activ_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class ActivationLayerTest : public LayerTestsCommon<activ_test_params> {
protected:
    std::string getModel(const activ_test_params& p) {
        std::map<std::string, std::string> params = {
                {"type", "sigmoid"}
        };

        InOutShapes inout = {{{p.in.n, p.in.c}},
                           {{p.in.n, p.in.c}}};

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Activation_Only", inout.inDims[0], p.precision)
                .addLayer("Activation", p.precision, &params, inout);
        return model.finish();
    }
};

class ReLULayerTest : public ActivationLayerTest {
protected:
    std::string getModel(const activ_test_params& p) {
        InOutShapes inout = {
                {{p.in.c, p.in.h, p.in.w}},
                {{p.in.c, p.in.h, p.in.w}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "ReLU_Only", inout.inDims[0], p.precision)
                .addLayer("ReLU", p.precision, nullptr, inout);
        return model.finish();
    }
};

class ClampLayerTest : public ActivationLayerTest {
protected:
    std::string getModel(const activ_test_params& p) {
        std::map<std::string, std::string> params = {
                {"min", "-50"},
                {"max", "50"}
        };

        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.h, p.in.w}},
                {{p.in.n, p.in.c, p.in.h, p.in.w}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Clamp_Only", inout.inDims[0], p.precision)
                .addLayer("Clamp", p.precision, &params, inout);
        return model.finish();
    }
};

TEST_P(ActivationLayerTest, CanNotLoadActivationLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported primitive of type: Activation name: Activation1";
    
    InferenceEngine::Core core;
    std::string model = getModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    if (param.device == CommonTestUtils::DEVICE_CPU) {
        ASSERT_NO_THROW(ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device));
    } else {
        try {
            ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
        } catch (InferenceEngineException ex) {
            ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
            ASSERT_STR_CONTAINS(ex.what(), ref_error);
        }
    }
}

TEST_P(ReLULayerTest, CanNotLoadReLULayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ? "Graph is not supported on FPGA" :
                       (param.device == CommonTestUtils::DEVICE_CPU)  ? "channels mismatch between mea" :
                       "Unsupported layer: ReLU1:ReLU";
    
    InferenceEngine::Core core;
    std::string model = getModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        if (param.device != CommonTestUtils::DEVICE_CPU) {
            ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        }
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }

}

TEST_P(ClampLayerTest, CanNotLoadClampLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Clamp1:Clamp";
    
    InferenceEngine::Core core;
    std::string model = getModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }

}

#define activation_case in_params_nchw({1, 96, 55, 55})
#define clamp_case in_params_nchw({1, 1, 512, 1})

struct norm_test_params : in_params_nchw, base_test_params {
    norm_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class NormalizeLayerTest : public LayerTestsCommon<norm_test_params> {
protected:
    std::string getModel(const norm_test_params& p) {
        std::map<std::string, std::string> params = {
                {"across_spatial", "0"},
                {"type",           "constant"},
                {"value",          "20.000000"},
                {"min",            "0.000000"},
                {"max",            "1.000000"},
                {"mean",           "0.000000"},
                {"std",            "1.000000"},
                {"sparse",         "-1"},
                {"variance_norm",  "caffe.FillerParameter.FAN_IN"},
                {"channel_shared", "0"},
                {"eps",            "0.000000"}
        };

        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.h, p.in.w}},
                {{p.in.n, p.in.c, p.in.h, p.in.w}}
        };
        size_t weights = 2048;

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Normalize_Only", inout.inDims[0], p.precision)
                .addLayer("Normalize", p.precision, &params, inout, weights);
        return model.finish();
    }

    TBlob<uint8_t>::Ptr GetNetworkWeights(const norm_test_params& p) {
        TBlob<uint8_t>* weights = new TBlob<uint8_t>(
                { Precision::U8, {p.in.c * sizeof(float)}, Layout::C });
        weights->allocate();
        fill_data(weights->buffer().as<float*>(),
                  weights->size() / sizeof(float));
        TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

        return weights_ptr;
    }
};

TEST_P(NormalizeLayerTest, CanNotLoadNormalizeLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Normalize1:Normalize";
    
    InferenceEngine::Core core;
    auto network = core.ReadNetwork(getModel(param), GetNetworkWeights(param));

    if (param.device == CommonTestUtils::DEVICE_CPU) {
        ASSERT_NO_THROW(ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device));
    } else {
        try {
            ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device);
        } catch (InferenceEngineException ex) {
            ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
            ASSERT_STR_CONTAINS(ex.what(), ref_error);
        }
    }
}

#define norm_case in_params_nchw({1, 512, 38, 38})

struct scale_params : in_params_nchw {
    size_t axis;

    scale_params(in_params_nchw in, size_t ax) :
            in_params_nchw(in) {
        axis = ax;
    }
};

struct scale_test_params : scale_params, base_test_params {
    scale_test_params(std::string name, std::string pr, scale_params params) :
            scale_params(params), base_test_params(name, pr) {}
};

class ScalingLayerTest : public LayerTestsCommon<scale_test_params> {
protected:
    std::string getScaleShiftModel(const scale_test_params& p) {
        InOutShapes inout = {
                {{p.in.w, p.in.h}},
                {{p.in.w, p.in.h}}
        };
        size_t weights = 2048;

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "ScaleShift_Only", inout.inDims[0], p.precision)
                .addLayer("ScaleShift", p.precision, nullptr, inout, weights);
        return model.finish();
    }

    std::string getSoftMaxModel(const scale_test_params& p) {
        std::map<std::string, std::string> params = {
                {"axis", std::to_string(p.axis)}
        };

        InOutShapes inout = {
                {{p.in.w, p.in.h}},
                {{p.in.w, p.in.h}}
        };
        size_t weights = 2048;

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "SoftMax_Only", inout.inDims[0], p.precision)
                .addLayer("SoftMax", p.precision, &params, inout, weights);
        return model.finish();
    }

    std::string getBatchNormalizationModel(const scale_test_params& p) {
        std::map<std::string, std::string> params = {
                {"epsilon", "2e-05"}
        };

        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.c, p.in.w, p.in.h}}
        };
        size_t weights = 12;
        size_t biases = 12;

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "BatchNormalization_Only", inout.inDims[0], p.precision)
                .addLayer("BatchNormalization", p.precision, &params, inout, weights, biases);
        return model.finish();
    }
};

TEST_P(ScalingLayerTest, CanNotLoadScaleShiftLayer) {
    auto param = GetParam();
    string ref_error = "Unsupported layer: ScaleShift1:ScaleShift";
    std::map<std::string, std::string> config;
    if (param.device == CommonTestUtils::DEVICE_FPGA) {
        ref_error = "Graph is not supported on FPGA";
    } else if (param.device == CommonTestUtils::DEVICE_GNA) {
        config.insert({GNA_CONFIG_KEY(SCALE_FACTOR), std::to_string(1)});
        ref_error = "[GNAPlugin] in function operator(): "
                "Incorrect weight value for ScaleShift1:ScaleShift";
    }
    
    InferenceEngine::Core core;
    std::string model = getScaleShiftModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, config);
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

TEST_P(ScalingLayerTest, CanNotLoadSoftMaxLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ? "Graph is not supported on FPGA" :
                       (param.device == CommonTestUtils::DEVICE_CPU) ? "Incorrect axis!" : "Unsupported layer: SoftMax1:SoftMax";
    
    InferenceEngine::Core core;
    std::string model = getSoftMaxModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device);
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

TEST_P(ScalingLayerTest, CanNotLoadBatchNormalizationLayer) {
    auto param = GetParam();
    string ref_error = "Unsupported layer: BatchNormalization1:BatchNormalization";

    if (param.device == CommonTestUtils::DEVICE_FPGA) {
        ref_error = "Graph is not supported on FPGA";
    } else if (param.device == CommonTestUtils::DEVICE_CPU) {
        ref_error = "Weights/biases are empty for layer: BatchNormalization1 ";
    } else if (param.device == CommonTestUtils::DEVICE_GNA) {
        ref_error = "[GNAPlugin] in function LoadNetwork: "
                "The plugin does not support layer: "
                "BatchNormalization1:BatchNormalization";
    }

    InferenceEngine::Core core;
    std::string model = getBatchNormalizationModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

struct shaping_test_params : in_params_nchw, base_test_params {
    shaping_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class ShapingLayerTest : public LayerTestsCommon<shaping_test_params> {
protected:
    std::string getFlattenModel(const shaping_test_params& p) {
        std::map<std::string, std::string> params = {
                {"axis",     "1"},
                {"end_axis", "-1"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.c}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Flatten_Only", inout.inDims[0], p.precision)
                .addLayer("Flatten", p.precision, &params, inout);
        return model.finish();
    }

    std::string getReshapeModel(const shaping_test_params& p) {
        std::map<std::string, std::string> params = {
                {"dim",      std::to_string(p.in.n) + "," + std::to_string(p.in.c)},
                {"axis",     "0"},
                {"num_axes", "-1"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.c}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Reshape_Only", inout.inDims[0], p.precision)
                .addLayer("Reshape", p.precision, &params, inout);
        return model.finish();
    }

    std::string getCropModel(const shaping_test_params& p) {
        std::map<std::string, std::string> params = {
                {"dim",    "12,12"},
                {"axis",   "2,3"},
                {"offset", "0,0"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.c, 12,     12}}
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Crop_Only", inout.inDims[0], p.precision)
                .addLayer("Crop", p.precision, &params, inout);
        return model.finish();
    }
};

TEST_P(ShapingLayerTest, CanNotLoadFlattenLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Flatten1:Flatten";
    
    InferenceEngine::Core core;
    std::string model = getFlattenModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

TEST_P(ShapingLayerTest, CanNotLoadReshapeLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Reshape1:Reshape";
    
    InferenceEngine::Core core;
    std::string model = getFlattenModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

TEST_P(ShapingLayerTest, CanNotLoadCropLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Crop1:Crop";
    
    InferenceEngine::Core core;
    std::string model = getFlattenModel(param);
    CNNNetwork network = core.ReadNetwork(model, Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

#define shape_case in_params_nchw({1, 512, 1, 1})

struct element_test_params : in_params_nchw, base_test_params {
    element_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class ElementWiseLayerTest : public LayerTestsCommon<element_test_params> {
protected:
    std::string getEltwiseModel(const element_test_params& p) {
        std::vector<size_t> dims{p.in.n, p.in.c};
        InOutShapes inout = {{dims, dims},
                           {dims}};

        std::map<std::string, std::string> params = {
                {"operation", "prod"}
        };

        return V2NetBuilder::buildNetworkWithOneInput(
                "Eltwise_Only", dims, p.precision)
                .addInputLayer(p.precision, dims)
                .addLayer("Eltwise", p.precision, &params, inout)
                .havingEdges().connect(0, 2).connect(1, 2).finish();
    }
};

TEST_P(ElementWiseLayerTest, CanNotLoadEltwiseLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Eltwise1:Eltwise";

    std::string model = getEltwiseModel(param);
    InferenceEngine::Core core;
    auto network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

struct object_test_params : in_params_nchw, base_test_params {
    object_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class ObjectDetectionLayerTest : public LayerTestsCommon<object_test_params> {
protected:
    std::string getPermuteModel(const object_test_params& p) {
        std::map<std::string, std::string> params = {
                {"order", "0,2,3,1"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.w, p.in.h, p.in.c}},
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "Permute_Only", inout.inDims[0], p.precision)
                .addLayer("Permute", p.precision, &params, inout);
        return model.finish();
    }

    std::string getPriorBoxModel(const object_test_params& p) {
        std::map<std::string, std::string> params = {
                {"min_size",     "162.000000"},
                {"max_size",     "213.000000"},
                {"aspect_ratio", "2.000000,3.000000"},
                {"flip",         "1"},
                {"clip",         "0"},
                {"variance",     "0.100000,0.100000,0.200000,0.200000"},
                {"img_size",     "0"},
                {"img_h",        "0"},
                {"img_w",        "0"},
                {"step",         "64.000000"},
                {"step_h",       "0.000000"},
                {"step_w",       "0.000000"},
                {"offset",       "0.500000"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c, p.in.w, p.in.h}},
                {{p.in.n, p.in.w, p.in.h, p.in.c}},
        };

        V2NetBuilder model = V2NetBuilder::buildNetworkWithOneInput(
                "PriorBox_Only", inout.inDims[0], p.precision)
                .addLayer("PriorBox", p.precision, &params, inout);
        return model.finish();
    }

    // TODO: add DetectionOutput and Tile layers
};

TEST_P(ObjectDetectionLayerTest, CanNotLoadPermuteLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Permute1:Permute";
    
    InferenceEngine::Core core;
    CNNNetwork network = core.ReadNetwork(getPermuteModel(param), Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device);
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}

#define scale_case scale_params({1, 512, 38, 38}, 2)

#define object_case in_params_nchw({1, 804, 38, 38})

struct memory_test_params : in_params_nchw, base_test_params {
    memory_test_params(std::string name, std::string pr, in_params_nchw params) :
            in_params_nchw(params), base_test_params(name, pr) {}
};

class MemoryLayerTest : public LayerTestsCommon<memory_test_params> {
protected:
    std::string getMemoryModel(const memory_test_params& p) {
        std::map<std::string, std::string> params = {
                {"id",    "r_2-3"},
                {"index", "1"},
                {"size",  "2"}
        };
        std::map<std::string, std::string> paramsFC = {
                {"out-size", "2048"}
        };
        InOutShapes inout = {
                {{p.in.n, p.in.c}},
                {{p.in.n, 2048}}
        };
        InOutShapes inoutMemory = {
                {{p.in.n, 2048}},
                {}
        };
        return V2NetBuilder::buildNetworkWithOneInput(
                "FC_with_Memory", inout.inDims[0], p.precision)
                .addInputLayer(p.precision, inout.inDims[0])
                .addLayer("FullyConnected", p.precision, &paramsFC, inout, 1638400)
                .addLayer("FullyConnected", p.precision, &paramsFC, inout, 1638400)
                .addLayer("Memory", p.precision, &paramsFC, inoutMemory)
                .havingEdges().connect(0, 2).connect(1, 3).connect(2, 4).finish();
    }
};

TEST_P(MemoryLayerTest, CanNotLoadMemoryLayer) {
    auto param = GetParam();
    string ref_error = (param.device == CommonTestUtils::DEVICE_FPGA) ?
                       "Graph is not supported on FPGA" : "Unsupported layer: Memory1:Memory";

    InferenceEngine::Core core;
    std::string model = getMemoryModel(param);
    auto network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr());

    try {
        ExecutableNetwork exeNetwork = core.LoadNetwork(network, param.device, {});
    } catch (InferenceEngineException ex) {
        ASSERT_EQ(ex.getStatus(), StatusCode::GENERAL_ERROR);
        ASSERT_STR_CONTAINS(ex.what(), ref_error);
    }
}
IE_SUPPRESS_DEPRECATED_END

#define memory_case in_params_nchw({1, 512, 38, 38})

