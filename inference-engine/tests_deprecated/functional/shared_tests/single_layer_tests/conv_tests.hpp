// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <details/ie_cnn_network_iterator.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <single_layer_common.hpp>
#include <string>
#include "conv_ref.hpp"
#include "common_test_utils/common_layers_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using std::vector;

struct conv_base_params {
    vector<size_t> in_dims;
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;
    vector<size_t> dilations;

    size_t out_c;
    size_t grp_c;

    vector<size_t> out_dims;
};

struct conv_test_params : conv_base_params {
    std::string device_name;

    std::string getDeviceName() const {
        return device_name;
    }
    conv_test_params(std::string name, conv_base_params params) :
            conv_base_params(params), device_name(name) {}
};

class ConvolutionOnlyTest : public TestsCommon,
                            public WithParamInterface<conv_test_params> {

    std::string model_t_4D = R"V0G0N(
<net name="Convolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution strides="_KS_"
                         pads_begin="_PB_" pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         output="_OC_" group="_GC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string model_t_4D_blobs_as_inputs = R"V0G0N(
<net name="Convolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="wei" type="Const" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>_OC_</dim>
                    <dim>_ICG_</dim>
                    <dim>_KH_</dim>
                    <dim>_KW_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="_S1_"/>
            </blobs>
        </layer>
        <layer name="bias" type="Const" precision="FP32" id="2">
            <output>
                <port id="0">
                    <dim>_OC_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_S1_" size="_S2_"/>
            </blobs>
        </layer>
        <layer name="conv1" id="3" type="Convolution" precision="FP32">
            <convolution strides="_KS_"
                         pads_begin="_PB_" pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         output="_OC_" group="_GC_"/>

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
                <port id="2">
                    <dim>_OC_</dim>
                    <dim>_ICG_</dim>
                    <dim>_KH_</dim>
                    <dim>_KW_</dim>
                </port>
                <port id="3">
                    <dim>_OC_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
    </edges>
</net>
)V0G0N";

    std::string model_t_5D = R"V0G0N(
<net name="Convolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution strides="_KS_"
                         pads_begin="_PB_"  pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         output="_OC_"  group="_GC_"/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OD_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string model_t_5D_blobs_as_inputs = R"V0G0N(
<net name="Convolution_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="wei" type="Const" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>_OC_</dim>
                    <dim>_ICG_</dim>
                    <dim>_KD_</dim>
                    <dim>_KH_</dim>
                    <dim>_KW_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="_S1_"/>
            </blobs>
        </layer>
        <layer name="bias" type="Const" precision="FP32" id="2">
            <output>
                <port id="0">
                    <dim>_OC_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_S1_" size="_S2_"/>
            </blobs>
        </layer>
        <layer name="conv1" id="3" type="Convolution" precision="FP32">
            <convolution strides="_KS_"
                         pads_begin="_PB_" pads_end="_PE_"
                         kernel="_K_"
                         dilations="_DL_"
                         output="_OC_" group="_GC_"/>

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_ID_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
                <port id="2">
                    <dim>_OC_</dim>
                    <dim>_ICG_</dim>
                    <dim>_KD_</dim>
                    <dim>_KH_</dim>
                    <dim>_KW_</dim>
                </port>
                <port id="3">
                    <dim>_OC_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>
                    <dim>_OD_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
    </edges>
</net>
)V0G0N";

protected:

    virtual bool blobsAsInputs() { return false; }

    size_t calculateOutDim(size_t in_dim, size_t kernel, size_t stride, size_t pad_begin) {
        return (in_dim + 2lu * pad_begin - kernel) / stride + 1lu;
    }

    void createBlobs(const conv_test_params &p, TBlob<float>::Ptr &src, TBlob<float>::Ptr &dst, TBlob<float>::Ptr &dst_ref) {
        auto in_size = p.in_dims.size();
        auto out_size = p.out_dims.size();
        SizeVector dims_dst = {
                p.out_dims[out_size - 1] == 0 ?
                calculateOutDim(p.in_dims[in_size - 1], p.kernel[X_AXIS], p.strides[X_AXIS], p.pads_begin[X_AXIS]) : p.out_dims[out_size - 1],
                p.out_dims[out_size - 2] == 0 ?
                calculateOutDim(p.in_dims[in_size - 2], p.kernel[Y_AXIS], p.strides[Y_AXIS], p.pads_begin[Y_AXIS]) : p.out_dims[out_size - 2],
                p.out_c,
                1lu};
        SizeVector dims_src;
        for (int i = in_size; i > 0; i--) {
            dims_src.push_back(p.in_dims[i - 1]);
        }

        Layout layout = NCHW;
        if (in_size == 5) {
            layout = NCDHW;
            dims_dst.insert(dims_dst.begin() + 2, p.out_dims.size() > 2 ?
                                                  (p.out_dims[out_size - 3] == 0 ?
                                                   calculateOutDim(p.in_dims[in_size - 3], p.kernel[Z_AXIS], p.strides[Z_AXIS], p.pads_begin[Z_AXIS]) : p.out_dims[out_size - 3]) : 1lu);
        }

        src = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_src.rbegin(), dims_src.rend()), layout));
        src->allocate();

        dst = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), layout));
        dst->allocate();

        dst_ref = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector(dims_dst.rbegin(), dims_dst.rend()), layout));
        dst_ref->allocate();
    }

    TBlob<uint8_t>::Ptr fillWeights(const conv_test_params &p) {
        auto KZ = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;
        TBlob<uint8_t> *weights_ptr = new TBlob<uint8_t>(TensorDesc(Precision::U8,
                                                                    {(p.kernel[X_AXIS] * p.kernel[Y_AXIS] * KZ * p.out_c * p.in_dims[1] / p.grp_c + p.out_c)
                                                                     * sizeof(float)}, C));
        weights_ptr->allocate();
        fill_data((float *) weights_ptr->buffer(), weights_ptr->size() / sizeof(float));
        return TBlob<uint8_t>::Ptr(weights_ptr);
    }

    void calculateRef(const TBlob<uint8_t>::Ptr &weights, const conv_test_params &p, const TBlob<float>::Ptr &src,
                      TBlob<float>::Ptr &dst_ref) {
        const float *weights_data = (const float *) weights->buffer();
        size_t bias_size = p.out_c;
        size_t weights_size = weights->size() / sizeof(float) - bias_size;
        const float *bias_data = weights_data + weights_size;
        CommonTestUtils::conv_common_params params;
        for (int i = 0; i < p.kernel.size(); i++)
            params.kernel.insert(i, p.kernel[i]);
        for (int i = 0; i < p.strides.size(); i++)
            params.stride.insert(i, p.strides[i]);
        for (int i = 0; i < p.pads_begin.size(); i++)
            params.pads_begin.insert(i, p.pads_begin[i]);
        for (int i = 0; i < p.dilations.size(); i++)
            params.dilation.insert(i, p.dilations[i]);
        params.group = p.grp_c;
        params.out_c = p.out_c;
        ref_conv_common<float>({ src }, *dst_ref.get(), weights_data, weights_size, bias_data, bias_size, params);
    }

    CNNNetwork getNetwork(const TBlob<uint8_t>::Ptr &weights, const conv_test_params &p) {
        Core ie;
        return ie.ReadNetwork(getModel(p), weights);
    }

    virtual void infer(CNNNetwork &network, const conv_test_params &p, TBlob<float>::Ptr &src, TBlob<float>::Ptr &dst) {
        Core ie;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.getDeviceName());
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        OutputsDataMap outInfo;
        outInfo = network.getOutputsInfo();
        inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
        inferRequest.SetBlob(outInfo.begin()->first, dst);
        inferRequest.Infer();
    }

    void SetUp() override {
        try {
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            TBlob<float>::Ptr src, dst, dst_ref;
            createBlobs(p, src, dst, dst_ref);
            fill_data(src->data(), src->size());
            auto weights = fillWeights(p);
            calculateRef(weights, p, src, dst_ref);
            CNNNetwork network = getNetwork(weights, p);
            infer(network, p, src, dst);
            compare(*dst, *dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

    virtual std::string getModel(conv_test_params p) {
        std::string model;
        auto in_dims_size = p.in_dims.size();
        if (in_dims_size == 4) {
            if (blobsAsInputs())
                model = model_t_4D_blobs_as_inputs;
            else
                model = model_t_4D;
        } else if (in_dims_size == 5) {
            if (blobsAsInputs())
                model = model_t_5D_blobs_as_inputs;
            else
                model = model_t_5D;
        }

        auto out_dims_size = p.out_dims.size();

        size_t KD = p.kernel.size() > Z_AXIS ? p.kernel[Z_AXIS] : 1lu;
        size_t KH = p.kernel[Y_AXIS];
        size_t KW = p.kernel[X_AXIS];

        size_t SD = p.strides.size() > Z_AXIS ? p.strides[Z_AXIS] : 1lu;
        size_t SH = p.strides[Y_AXIS];
        size_t SW = p.strides[X_AXIS];

        size_t ID = p.in_dims.size() > 4 ? p.in_dims[in_dims_size - 3] : 1lu;
        size_t IH = p.in_dims[in_dims_size - 2];
        size_t IW = p.in_dims[in_dims_size - 1];

        size_t OD = p.out_dims.size() > 2 ? p.out_dims[out_dims_size - 3] : 1lu;
        size_t OH = p.out_dims[out_dims_size - 2];
        size_t OW = p.out_dims[out_dims_size - 1];

        size_t PD = p.pads_begin.size() > Z_AXIS ? p.pads_begin[Z_AXIS] : 1lu;
        size_t PH = p.pads_begin[Y_AXIS];
        size_t PW = p.pads_begin[X_AXIS];

        REPLACE_WITH_NUM(model, "_IW_", IW);
        REPLACE_WITH_NUM(model, "_IH_", IH);
        REPLACE_WITH_NUM(model, "_ID_", ID);
        REPLACE_WITH_NUM(model, "_IC_", p.in_dims[1]);
        REPLACE_WITH_NUM(model, "_ICG_", p.in_dims[1] / p.grp_c);
        REPLACE_WITH_NUM(model, "_IN_", p.in_dims[0]);

        REPLACE_WITH_NUM(model, "_KD_", KD);
        REPLACE_WITH_NUM(model, "_KH_", KH);
        REPLACE_WITH_NUM(model, "_KW_", KW);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_end);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_DL_", p.dilations);

        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);
        REPLACE_WITH_NUM(model, "_OD_", out_dims_size > 2 ? (OD == 0 ? calculateOutDim(ID, KD, SD, PD) : OD) : 1lu);
        REPLACE_WITH_NUM(model, "_OH_", OH == 0 ? calculateOutDim(IH, KH, SH, PH) : OH);
        REPLACE_WITH_NUM(model, "_OW_", OW == 0 ? calculateOutDim(IW, KW, SW, PW) : OW);

        size_t w_data_size = (KW * KH * KD * p.out_c * p.in_dims[1] / p.grp_c) * sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);
        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);
        return model;
    }
};

class ConvolutionReshapeTest : public ConvolutionOnlyTest {
protected:
    void SetUp() override {
        try {
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            TBlob<float>::Ptr src, dst, dst_ref;
            createBlobs(p, src, dst, dst_ref);
            fill_data(src->data(), src->size());
            auto weights = fillWeights(p);
            calculateRef(weights, p, src, dst_ref);
            CNNNetwork network = getNetwork(weights, p);
            updatePaddings(network, p);
            infer(network, p, src, dst);
            compare(*dst, *dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

    void updatePaddings(const CNNNetwork &network, conv_test_params& p) {
        details::CNNNetworkIterator i(network), end;
        auto found = std::find_if(i, end, [](const CNNLayer::Ptr& layer) {
            return layer->type == "Convolution";
        });
        ASSERT_NE(found, end);
        auto convLayer = std::dynamic_pointer_cast<ConvolutionLayer>(*found);
        auto allPad = getPaddings(*convLayer.get());
        p.pads_begin[X_AXIS] = allPad.begin[X_AXIS];
        p.pads_begin[Y_AXIS] = allPad.begin[Y_AXIS];
        if (p.pads_begin.size() > Z_AXIS)
            p.pads_begin[Z_AXIS] = allPad.begin[Z_AXIS];
    }

    void infer(CNNNetwork &network, const conv_test_params &p, TBlob<float>::Ptr &src, TBlob<float>::Ptr &dst) override {
        Core ie;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.getDeviceName());
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        OutputsDataMap outInfo;
        outInfo = network.getOutputsInfo();
        inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
        inferRequest.SetBlob(outInfo.begin()->first, dst);
        inferRequest.Infer();
    }

    std::string getModel(conv_test_params p) override {
        std::string model = ConvolutionOnlyTest::getModel(p);
        REPLACE_WITH_STR(model, "convolution", "convolution auto_pad=\"same_upper\"");
        std::string pads_pattern = "pads_begin=\"";
        for (int i = p.pads_begin.size(); i > 0; i--) {
            pads_pattern += std::to_string(p.pads_begin[i - 1]) + ",";
        }
        auto end = pads_pattern.end()--;
        *end = '\"';
        std::string pads = "pads_begin=\"0,0\"";
        if (p.pads_begin.size() == 3) {
            pads = "pads_begin=\"0,0,0\"";
        }
        REPLACE_WITH_NUM_VECTOR(model, pads_pattern, pads);
        return model;
    }
};


class ConvolutionBlobsAsInputsTest : public ConvolutionOnlyTest {
protected:
    bool blobsAsInputs() override { return true; }
};

#define case_1  conv_base_params({{1lu, 9lu, 16lu, 32lu},  {1lu, 1lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}})
#define case_2  conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}})
#define case_3  conv_base_params({{1lu, 9lu, 32lu, 16lu},  {2lu, 4lu}, {2lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}})
#define case_4  conv_base_params({{1lu, 3lu, 40lu, 40lu},  {3lu, 3lu}, {1lu, 2lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 20lu, 1lu, {0lu, 0lu}})
#define case_5  conv_base_params({{1lu, 9lu, 16lu, 32lu},  {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 17lu, 1lu, {0lu, 0lu}})
#define case_6  conv_base_params({{1lu, 3lu, 224lu, 224lu}, {7lu, 7lu}, {2lu, 2lu}, {2lu, 2lu}, {0lu, 0lu}, {1lu, 1lu}, 64lu, 1lu, {112lu, 112lu}})
#define case_7  conv_base_params({{1lu, 16lu, 40lu, 40lu}, {3lu, 3lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {1lu, 1lu}, 16lu, 16lu, {0lu, 0lu}})
#define case_8  conv_base_params({{1lu, 32lu, 16lu, 32lu}, {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {1lu, 1lu}, 32lu, 32lu, {0lu, 0lu}})
#define case_9  conv_base_params({{1lu, 16lu, 40lu, 40lu}, {3lu, 3lu}, {1lu, 1lu}, {0lu, 0lu}, {0lu, 0lu}, {9lu, 9lu}, 16lu, 16lu, {0lu, 0lu}})
#define case_10 conv_base_params({{1lu, 32lu, 16lu, 32lu}, {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {9lu, 9lu}, 32lu, 32lu, {0lu, 0lu}})
#define case_11 conv_base_params({{1lu, 4lu, 16lu, 32lu},  {7lu, 7lu}, {2lu, 2lu}, {3lu, 3lu}, {0lu, 0lu}, {9lu, 9lu}, 4lu, 4lu, {0lu, 0lu}})
#define case_12 conv_base_params({{1lu, 3lu, 224lu, 224lu}, {10lu, 10lu}, {1lu, 1lu}, {4lu, 4lu}, {0lu, 0lu}, {1lu, 1lu}, 4lu, 1lu, {224lu, 224lu}})

#define case_13  conv_base_params({{1lu, 3lu, 16lu, 32lu, 32lu},  {1lu, 1lu, 1lu}, {1lu, 1lu, 1lu}, {0lu, 0lu, 0lu}, {0lu, 0lu, 0lu}, {1lu, 1lu, 1lu}, 17lu, 1lu, {0lu, 0lu, 0lu}})
#define case_14  conv_base_params({{1lu, 3lu, 16lu, 32lu, 32lu},  {3lu, 3lu, 3lu}, {2lu, 2lu, 1lu}, {0lu, 0lu, 0lu}, {0lu, 0lu, 0lu}, {1lu, 1lu, 1lu}, 64lu, 1lu, {0lu, 0lu, 0lu}})

// NOTE: always auto_pad = same_upper. IR with zero_pads, pad from params is used for ref_conv after reshape
#define case_si_1 conv_base_params({{1lu, 144lu, 75lu, 75lu}, {3lu, 3lu}, {2lu, 2lu}, {1lu, 1lu}, {0lu, 0lu}, {1lu, 1lu}, 144lu, 144lu, {1lu, 1lu}})

TEST_P(ConvolutionOnlyTest, TestsConvolution) {
}

TEST_P(ConvolutionReshapeTest, TestsReshapeConvolution) {
}

TEST_P(ConvolutionBlobsAsInputsTest, TestsConvolutionBlobsAsInputs) {
}

std::string getTestCaseName(testing::TestParamInfo<conv_test_params> obj) {
    auto in_dims_size = obj.param.in_dims.size();
    return obj.param.device_name +
           "_w" + std::to_string(obj.param.in_dims[in_dims_size - 1]) +
           "_h" + std::to_string(obj.param.in_dims[in_dims_size - 2]) +
           (obj.param.in_dims.size() > 4 ? "_d" + std::to_string(obj.param.in_dims[in_dims_size - 3]) : "") +
           "_c" + std::to_string(obj.param.in_dims[1]) +
           "_kw" + std::to_string(obj.param.kernel[X_AXIS]) +
           "_kh" + std::to_string(obj.param.kernel[Y_AXIS]) +
           (obj.param.kernel.size() > Z_AXIS ? "_kd" + std::to_string(obj.param.kernel[Z_AXIS]) : "") +
           "_sw" + std::to_string(obj.param.strides[X_AXIS]) +
           "_sh" + std::to_string(obj.param.strides[Y_AXIS]) +
           (obj.param.strides.size() > Z_AXIS ? "_sd" + std::to_string(obj.param.strides[Z_AXIS]) : "") +
           "_dilw" + std::to_string(obj.param.dilations[X_AXIS]) +
           "_dilh" + std::to_string(obj.param.dilations[Y_AXIS]) +
           (obj.param.dilations.size() > Z_AXIS ? "_dild" + std::to_string(obj.param.dilations[Z_AXIS]) : "") +
           "_grpc" + std::to_string(obj.param.grp_c);
}
