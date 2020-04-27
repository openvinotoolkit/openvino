// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct detectionout_test_params {
    std::string device_name;

    size_t mb;

    struct {
        size_t c;
    } in1;

    struct {
        size_t c;
    } in2;

    struct {
        size_t c;
        size_t h;
        size_t w;
    } in3;

    struct {
        size_t c;
        size_t h;
        size_t w;
    } out;
};

class smoke_CPUDetectionOutOnlyTest: public TestsCommon,
                             public WithParamInterface<detectionout_test_params> {

    std::string model_t = R"V0G0N(
<Net Name="PriorBox_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                </port>
            </output>
        </layer>
        <layer name="input2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                </port>
            </output>
        </layer>
        <layer name="input3" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>_IC3_</dim>
                    <dim>_IH3_</dim>
                    <dim>_IW3_</dim>
                </port>
            </output>
        </layer>
        <layer name="detection_out" type="DetectionOutput" precision="FP32" id="11">
            <data num_classes="4" share_location="1" background_label_id="0" nms_threshold="0.400000" top_k="400"
                  output_directory="" output_name_prefix="" output_format="" label_map_file=""
                  name_size_file="" num_test_image="0" code_type="caffe.PriorBoxParameter.CENTER_SIZE"
                  variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"
                  visualize="0" visualize_threshold="0.000000" num_orient_classes="8"
                  interpolate_orientation="1" clip="1" decrease_label_id="1" />
            <input>
                <port id="11">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                </port>
                <port id="12">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                </port>
                <port id="13">
                    <dim>1</dim>
                    <dim>_IC3_</dim>
                    <dim>_IH3_</dim>
                    <dim>_IW3_</dim>
                </port>
            </input>
            <output>
                <port id="14">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="11" to-port="11"/>
        <edge from-layer="2" from-port="2" to-layer="11" to-port="12"/>
        <edge from-layer="3" from-port="3" to-layer="11" to-port="13"/>
    </edges>

</Net>
)V0G0N";

    std::string getModel(detectionout_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);
        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);

        REPLACE_WITH_NUM(model, "_IC3_", p.in3.c);
        REPLACE_WITH_NUM(model, "_IH3_", p.in3.h);
        REPLACE_WITH_NUM(model, "_IW3_", p.in3.w);

        REPLACE_WITH_NUM(model, "_OC_", p.out.c);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OW_", p.out.w);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            detectionout_test_params p = ::testing::WithParamInterface<detectionout_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());
            network.setBatchSize(p.mb);

            InputsDataMap inputs = network.getInputsInfo();

            DataPtr inputPtr1 = inputs["input1"]->getInputData();
            DataPtr inputPtr2 = inputs["input2"]->getInputData();
            DataPtr inputPtr3 = inputs["input3"]->getInputData();

            InferenceEngine::Blob::Ptr input1 = InferenceEngine::make_shared_blob<float>(inputPtr1->getTensorDesc());
            input1->allocate();

            InferenceEngine::Blob::Ptr input2 = InferenceEngine::make_shared_blob<float>(inputPtr2->getTensorDesc());
            input2->allocate();

            InferenceEngine::Blob::Ptr input3 = InferenceEngine::make_shared_blob<float>(inputPtr3->getTensorDesc());
            input3->allocate();

            InferenceEngine::BlobMap inputBlobs;
            inputBlobs["input1"] = input1;
            inputBlobs["input2"] = input2;
            inputBlobs["input3"] = input3;

            OutputsDataMap outputs = network.getOutputsInfo();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(outputs["detection_out"]->getTensorDesc());
            output->allocate();

            InferenceEngine::BlobMap outputBlobs;
            outputBlobs["detection_out"] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(inputBlobs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUDetectionOutOnlyTest, TestsDetectionOut) {}

INSTANTIATE_TEST_CASE_P(
        TestsDetectionOut, smoke_CPUDetectionOutOnlyTest,
        ::testing::Values(
                detectionout_test_params{ "CPU",
                    10, {147264}, {147264}, {2, 1, 147264}, {1, 200, 7} }));
