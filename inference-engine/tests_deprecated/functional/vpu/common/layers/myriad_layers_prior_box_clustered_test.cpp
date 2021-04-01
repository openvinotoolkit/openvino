// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

struct PriorBoxClusteredParams {
    tensor_test_params in1 = {1, 384, 19, 19};
    tensor_test_params in2 = {1, 3, 300, 300};

    std::vector<float> widths = {9.4f, 25.1f, 14.7f, 34.7f, 143.0f, 77.4f, 128.8f, 51.1f, 75.6f};
    std::vector<float> heights = {15.0f, 39.6f, 25.5f, 63.2f, 227.5f, 162.9f, 124.5f, 105.1f, 72.6f};
    int clip = 0;
    std::vector<float> variance = {0.1f, 0.1f, 0.2f, 0.2f};
    int img_h = 0;
    int img_w = 0;
    float step = 16.0;
    float step_h = 0.0;
    float step_w = 0.0;
    float offset = 0.5;
};

void refPriorBoxClustered(Blob::Ptr dst, const PriorBoxClusteredParams &p) {
    int num_priors = p.widths.size();

    int layer_width  = p.in1.w;
    int layer_height = p.in1.h;

    int32_t img_width  = p.img_w == 0 ? p.in2.w : p.img_w;
    int32_t img_height = p.img_h == 0 ? p.in2.h : p.img_h;

    float step_w = p.step_w == 0 ? p.step : p.step_w;
    float step_h = p.step_h == 0 ? p.step : p.step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }

    int offset = dst->getTensorDesc().getDims().back();
    int var_size = p.variance.size();

    ie_fp16* top_data_0 = static_cast<ie_fp16*>(dst->buffer());
    ie_fp16* top_data_1 = top_data_0 + offset;

    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width;  ++w) {
            float center_x = (w + p.offset) * step_w;
            float center_y = (h + p.offset) * step_h;

            for (int s = 0; s < num_priors; ++s) {
                float box_width  = p.widths[s];
                float box_height = p.heights[s];

                float xmin = (center_x - box_width  / 2.) / img_width;
                float ymin = (center_y - box_height / 2.) / img_height;
                float xmax = (center_x + box_width  / 2.) / img_width;
                float ymax = (center_y + box_height / 2.) / img_height;

                if (p.clip) {
                    xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                    ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                    xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                    ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                }

                top_data_0[h * layer_width * num_priors * 4 + w * num_priors * 4 + s * 4 + 0] = PrecisionUtils::f32tof16(xmin);
                top_data_0[h * layer_width * num_priors * 4 + w * num_priors * 4 + s * 4 + 1] = PrecisionUtils::f32tof16(ymin);
                top_data_0[h * layer_width * num_priors * 4 + w * num_priors * 4 + s * 4 + 2] = PrecisionUtils::f32tof16(xmax);
                top_data_0[h * layer_width * num_priors * 4 + w * num_priors * 4 + s * 4 + 3] = PrecisionUtils::f32tof16(ymax);

                for (int j = 0; j < var_size; j++) {
                    int index = h * layer_width * num_priors * var_size + w * num_priors * var_size + s * var_size + j;
                    top_data_1[index] = PrecisionUtils::f32tof16(p.variance[j]);
                }
            }
        }
    }
}

TEST_F(myriadLayersTests_nightly, PriorBoxClustered) {
    std::string model = R"V0G0N(
        <net name="PriorBoxClustered" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>384</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>384</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>384</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorboxclustered" type="PriorBoxClustered" precision="FP16" id="5">
                    <data
                        min_size="#"
                        max_size="#"
                        aspect_ratio="#"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="16.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="9.400000,25.100000,14.700000,34.700001,143.000000,77.400002,128.800003,51.099998,75.599998"
                        height="15.000000,39.599998,25.500000,63.200001,227.500000,162.899994,124.500000,105.099998,72.599998"/>
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>384</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>12996</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorboxclustered_copy" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="61">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>12996</dim>
                        </port>
                    </input>
                    <output>
                        <port id="62">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>12996</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
                <edge from-layer="5" from-port="53" to-layer="6" to-port="61"/>
            </edges>
        </net>
    )V0G0N";
    SetSeed(DEFAULT_SEED_VALUE + 6);
    PriorBoxClusteredParams params;

    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data1"]->setPrecision(Precision::FP16);
    _inputsInfo["data2"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["priorboxclustered_copy"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, {}));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
    
    Blob::Ptr data1;
    ASSERT_NO_THROW(data1 = _inferRequest.GetBlob("data1"));
    
    Blob::Ptr data2;
    ASSERT_NO_THROW(data2 = _inferRequest.GetBlob("data2"));
    
    GenRandomData(data1);
    GenRandomData(data2);

    ASSERT_NO_THROW(_inferRequest.Infer());
    
    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("priorboxclustered_copy"));
    
    _refBlob = make_shared_blob<ie_fp16>({Precision::FP16, outputBlob->getTensorDesc().getDims(), ANY});
    _refBlob->allocate();

    refPriorBoxClustered(_refBlob, params);

    CompareCommonAbsolute(outputBlob, _refBlob, 0.0);
}
