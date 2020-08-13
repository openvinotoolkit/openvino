// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct quantize_test_params {
    std::string device_name;

    struct {
        size_t n;
        size_t c;
        size_t h;
        size_t w;
    } in;

    size_t ic_const_blobs;
    size_t levels;
    bool reverse_out_vals;
};

template<typename data_t>
void ref_quantize(const std::vector<Blob::Ptr> &srcs, std::vector<Blob::Ptr> &dsts, quantize_test_params prm) {
    assert(dsts.size() == 1);

    const data_t* src_data = srcs[0]->buffer().as<data_t*>();
    const data_t* input_low_data = srcs[1]->buffer().as<data_t*>();
    const data_t* input_high_data = srcs[2]->buffer().as<data_t*>();
    const data_t* output_low_data = srcs[3]->buffer().as<data_t*>();
    const data_t* output_high_data = srcs[4]->buffer().as<data_t*>();

    data_t* dst_data = dsts[0]->buffer().as<data_t*>();

    size_t N = prm.in.n;
    size_t C = prm.in.c;
    size_t H = prm.in.h;
    size_t W = prm.in.w;
    size_t ICB = prm.ic_const_blobs;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    size_t idx = n*C*H*W + c*H*W + h*W + w;

                    if (src_data[idx] <= input_low_data[c % ICB])
                        dst_data[idx] = output_low_data[c % ICB];
                    else if (src_data[idx] > input_high_data[c % ICB])
                        dst_data[idx] = output_high_data[c % ICB];
                    else
                        dst_data[idx] = roundf((src_data[idx] - input_low_data[c % ICB]) /
                                               (input_high_data[c % ICB] - input_low_data[c % ICB]) * (prm.levels-1)) /
                                        (prm.levels-1) * (output_high_data[c % ICB] - output_low_data[c % ICB]) + output_low_data[c % ICB];
                }
            }
        }
    }
}

class QuantizeOnlyTest : public TestsCommon, public WithParamInterface<quantize_test_params> {

    std::string model_t = R"V0G0N(
<Net Name="Quantize_Only" version="6" precision="FP32" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="input_low" type="Const" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_O1_" size="_S1_"/>
            </blobs>
        </layer>
        <layer name="input_high" type="Const" precision="FP32" id="2">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_O2_" size="_S2_"/>
            </blobs>
        </layer>
        <layer name="output_low" type="Const" precision="FP32" id="3">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_O3_" size="_S3_"/>
            </blobs>
        </layer>
        <layer name="output_high" type="Const" precision="FP32" id="4">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_O4_" size="_S4_"/>
            </blobs>
        </layer>
        <layer name="quantize" type="FakeQuantize" precision="FP32" id="5">
            <data levels="_L_"/>
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>_ICB_</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
    </edges>
</Net>
)V0G0N";

    std::string getModel(quantize_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IN_",  p.in.n);
        REPLACE_WITH_NUM(model, "_IC_",  p.in.c);
        REPLACE_WITH_NUM(model, "_IH_",  p.in.h);
        REPLACE_WITH_NUM(model, "_IW_",  p.in.w);
        REPLACE_WITH_NUM(model, "_L_",   p.levels);
        REPLACE_WITH_NUM(model, "_ICB_", p.ic_const_blobs);

        REPLACE_WITH_NUM(model, "_O1_",  0 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_S1_",  1 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_O2_",  1 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_S2_",  1 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_O3_",  2 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_S3_",  1 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_O4_",  3 * p.ic_const_blobs * sizeof(float));
        REPLACE_WITH_NUM(model, "_S4_",  1 * p.ic_const_blobs * sizeof(float));

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            quantize_test_params p = ::testing::WithParamInterface<quantize_test_params>::GetParam();
            std::string model = getModel(p);

            std::vector<Blob::Ptr> srcs_vec;
            Blob::Ptr blob_data = make_shared_blob<float>({Precision::FP32, {p.in.n, p.in.c, p.in.h, p.in.w}, Layout::NCHW});
            blob_data->allocate();
            CommonTestUtils::fill_data_sine(blob_data->buffer().as<float*>(), blob_data->size(), 0.f, 2.f, 0.1f);
            srcs_vec.push_back(blob_data);

            float low_center = p.levels == 2 ? 0.f : -1.f;
            float high_center = p.levels == 2 ? 0.f : 1.f;
            float low_val = p.reverse_out_vals ? 1.0f : -1.f;
            float high_val = p.reverse_out_vals ? -1.0f : 1.f;

            Blob::Ptr input_low_data = make_shared_blob<float>({Precision::FP32, {p.ic_const_blobs}, Layout::C});
            input_low_data->allocate();
            CommonTestUtils::fill_data_sine(input_low_data->buffer().as<float*>(), input_low_data->size(), low_center, 2.f, 0.2f);
            srcs_vec.push_back(input_low_data);

            Blob::Ptr input_high_data = make_shared_blob<float>({Precision::FP32, {p.ic_const_blobs}, Layout::C});
            input_high_data->allocate();
            CommonTestUtils::fill_data_sine(input_high_data->buffer().as<float*>(), input_high_data->size(), high_center, 2.f, 0.2f);
            srcs_vec.push_back(input_high_data);

            Blob::Ptr output_low_data = make_shared_blob<float>({Precision::FP32, { p.ic_const_blobs }, Layout::C});
            output_low_data->allocate();
            if (p.levels == 2) {
                CommonTestUtils::fill_data_const(output_low_data, low_val);
            } else {
                CommonTestUtils::fill_data_sine(output_low_data->buffer().as<float*>(), output_low_data->size(), low_center, 2.f, 0.3f);
            };
            srcs_vec.push_back(output_low_data);

            Blob::Ptr output_high_data = make_shared_blob<float>({Precision::FP32, {p.ic_const_blobs}, Layout::C});
            output_high_data->allocate();
            if (p.levels == 2) {
                CommonTestUtils::fill_data_const(output_high_data, high_val);
            } else {
                CommonTestUtils::fill_data_sine(output_high_data->buffer().as<float*>(), output_high_data->size(), high_center, 2.f, 0.3f);
            };
            srcs_vec.push_back(output_high_data);

            TBlob<uint8_t> *weights_ptr = new TBlob<uint8_t>({Precision::U8, {4 * p.ic_const_blobs * sizeof(float)}, Layout::C});
            weights_ptr->allocate();

            float* pwei = weights_ptr->buffer().as<float*>();
            int off = 0;
            for (int i = 1; i < 5; i++) {
                float* pdata = srcs_vec[i]->buffer();
                for (int j = 0; j < p.ic_const_blobs; j++) {
                    pwei[off++] = pdata[j];
                }
            }

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, TBlob<uint8_t>::Ptr(weights_ptr));

            std::map<std::string, std::string> config = {{PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE, PluginConfigParams::NO}};
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name, config);
            InferRequest inferRequest = executable_network.CreateInferRequest();
            inferRequest.SetBlob("data", blob_data);

            std::vector<Blob::Ptr> dsts_vec;
            std::vector<Blob::Ptr> out_vec;

            OutputsDataMap out_info_map = net.getOutputsInfo();
            for (auto info : out_info_map) {
                Blob::Ptr blob = make_shared_blob<float>({Precision::FP32, info.second->getDims() , Layout::NCHW});
                blob->allocate();
                inferRequest.SetBlob(info.first, blob);
                out_vec.push_back(blob);

                Blob::Ptr blob_ref = make_shared_blob<float>({Precision::FP32, info.second->getDims(), Layout::NCHW});
                blob_ref->allocate();
                dsts_vec.push_back(blob_ref);
            }

            ref_quantize<float>(srcs_vec, dsts_vec, p);

            inferRequest.Infer();

            compare(*out_vec[0], *dsts_vec[0]);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

// {N, C, H, W}, ic_const_blobs, quantization_levels, reverse_out_vals
#define case_1 {1, 8, 5, 5}, 1, 2, false
#define case_2 {1, 8, 5, 5}, 8, 2, false
#define case_3 {1, 8, 5, 5}, 1, 4, false
#define case_4 {1, 8, 5, 5}, 8, 4, false
#define case_5 {1, 8, 5, 4}, 1, 8, false
#define case_6 {1, 8, 5, 4}, 8, 8, false
#define case_7 {1, 17, 5, 5}, 1, 2, false
#define case_8 {1, 17, 5, 5}, 17, 2, false
#define case_9 {1, 17, 5, 5}, 1, 4, false
#define case_10 {1, 17, 5, 5}, 17, 4, false
#define case_11 {1, 17, 5, 4}, 1, 8, false
#define case_12 {1, 17, 5, 4}, 17, 8, false
#define case_13 {1, 8, 5, 5}, 1, 2, true
#define case_14 {1, 8, 5, 5}, 8, 2, true
#define case_15 {1, 8, 5, 5}, 1, 4, true
#define case_16 {1, 8, 5, 5}, 8, 4, true
#define case_17 {1, 8, 5, 4}, 1, 8, true
#define case_18 {1, 8, 5, 4}, 8, 8, true
#define case_19 {1, 17, 5, 5}, 1, 2, true
#define case_20 {1, 17, 5, 5}, 17, 2, true
#define case_21 {1, 17, 5, 5}, 1, 4, true
#define case_22 {1, 17, 5, 5}, 17, 4, true
#define case_23 {1, 17, 5, 4}, 1, 8, true
#define case_24 {1, 17, 5, 4}, 17, 8, true

TEST_P(QuantizeOnlyTest, TestsQuantize) {}
