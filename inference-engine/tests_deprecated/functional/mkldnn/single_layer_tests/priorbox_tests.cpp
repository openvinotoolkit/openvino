// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct priorbox_test_params {
    std::string device_name;

    size_t mb;

    struct {
        size_t c;
        size_t h;
        size_t w;
    } in1;

    struct {
        size_t c;
        size_t h;
        size_t w;
    } in2;

    struct {
        size_t c;
        size_t h;
        size_t w;
    } out;

    int offset;
    int stride;
    int min_size;
    int max_size;
    bool flip;
    bool clip;
};

class smoke_CPUPriorBoxOnlyTest: public TestsCommon,
                             public WithParamInterface<priorbox_test_params> {

    std::string model_t = R"V0G0N(
<Net Name="PriorBox_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="input2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </output>
        </layer>
        <layer name="prior" type="PriorBox" precision="FP32" id="2">
            <data min_size="4.000000" max_size="9.000000" flip="1" clip="1" offset="0" step="0" aspect_ratio="" variance=""/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>

</Net>
)V0G0N";

    std::string getModel(priorbox_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW1_", p.in1.w);
        REPLACE_WITH_NUM(model, "_IH1_", p.in1.h);
        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);

        REPLACE_WITH_NUM(model, "_IW2_", p.in2.w);
        REPLACE_WITH_NUM(model, "_IH2_", p.in2.h);
        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);

        REPLACE_WITH_NUM(model, "_OW_", p.out.w);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OC_", p.out.c);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            priorbox_test_params p = ::testing::WithParamInterface<priorbox_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());
            network.setBatchSize(p.mb);

            InputsDataMap inputs = network.getInputsInfo();

            DataPtr inputPtr1 = inputs["input1"]->getInputData();
            DataPtr inputPtr2 = inputs["input2"]->getInputData();

            InferenceEngine::Blob::Ptr input1 = InferenceEngine::make_shared_blob<float>(inputPtr1->getTensorDesc());
            input1->allocate();

            InferenceEngine::Blob::Ptr input2 = InferenceEngine::make_shared_blob<float>(inputPtr2->getTensorDesc());
            input2->allocate();

            InferenceEngine::BlobMap inputBlobs;
            inputBlobs["input1"] = input1;
            inputBlobs["input2"] = input2;

            OutputsDataMap outputs = network.getOutputsInfo();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(outputs["prior"]->getTensorDesc());
            output->allocate();

            InferenceEngine::BlobMap outputBlobs;
            outputBlobs["prior"] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(inputBlobs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            // Check results

            const TBlob<float>::Ptr outputArray = std::dynamic_pointer_cast<TBlob<float>>(output);
            float* dst_ptr = outputArray->data();

            const float eps = 1e-6;

            // pick a few generated priors and compare against the expected number.
            // first prior
            EXPECT_NEAR(dst_ptr[0], 0.03, eps);
            EXPECT_NEAR(dst_ptr[1], 0.03, eps);
            EXPECT_NEAR(dst_ptr[2], 0.07, eps);
            EXPECT_NEAR(dst_ptr[3], 0.07, eps);
            // second prior
            EXPECT_NEAR(dst_ptr[4], 0.02, eps);
            EXPECT_NEAR(dst_ptr[5], 0.02, eps);
            EXPECT_NEAR(dst_ptr[6], 0.08, eps);
            EXPECT_NEAR(dst_ptr[7], 0.08, eps);
            // prior in the 5-th row and 5-th col
            EXPECT_NEAR(dst_ptr[4*10*2*4+4*2*4], 0.43, eps);
            EXPECT_NEAR(dst_ptr[4*10*2*4+4*2*4+1], 0.43, eps);
            EXPECT_NEAR(dst_ptr[4*10*2*4+4*2*4+2], 0.47, eps);
            EXPECT_NEAR(dst_ptr[4*10*2*4+4*2*4+3], 0.47, eps);

            // check variance
            dst_ptr += p.out.h * p.out.w;
            for (int d = 0; d < p.out.h * p.out.w; ++d) {
                EXPECT_NEAR(dst_ptr[d], 0.1, eps);
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUPriorBoxOnlyTest, TestsPriorBox) {}

INSTANTIATE_TEST_CASE_P(
        TestsPriorBox, smoke_CPUPriorBoxOnlyTest,
        ::testing::Values(
                priorbox_test_params{ "CPU",
                    10, {10, 10, 10}, {3, 100, 100}, {2, 1, 800}, 0, 0, 4, 9, true, true }));


class smoke_CPUPriorBoxDensityTest : public TestsCommon,
    public WithParamInterface<priorbox_test_params> {

    std::string model_t = R"V0G0N(
<Net Name="PriorBox_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="input2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </output>
        </layer>
        <layer name="prior" type="PriorBox" precision="FP32" id="2">
            <data fixed_size="4.000000" density="1.000000" flip="1" clip="1" offset="0" step="0" aspect_ratio="1.0" variance=""/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>

</Net>
)V0G0N";

    std::string getModel(priorbox_test_params p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW1_", p.in1.w);
        REPLACE_WITH_NUM(model, "_IH1_", p.in1.h);
        REPLACE_WITH_NUM(model, "_IC1_", p.in1.c);

        REPLACE_WITH_NUM(model, "_IW2_", p.in2.w);
        REPLACE_WITH_NUM(model, "_IH2_", p.in2.h);
        REPLACE_WITH_NUM(model, "_IC2_", p.in2.c);

        REPLACE_WITH_NUM(model, "_OW_", p.out.w);
        REPLACE_WITH_NUM(model, "_OH_", p.out.h);
        REPLACE_WITH_NUM(model, "_OC_", p.out.c);

        return model;
    }

protected:
    virtual void SetUp() {

        try {
            priorbox_test_params p = ::testing::WithParamInterface<priorbox_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork network = ie.ReadNetwork(model, Blob::CPtr());
            network.setBatchSize(p.mb);

            InputsDataMap inputs = network.getInputsInfo();

            DataPtr inputPtr1 = inputs["input1"]->getInputData();
            DataPtr inputPtr2 = inputs["input2"]->getInputData();

            InferenceEngine::Blob::Ptr input1 = InferenceEngine::make_shared_blob<float>(inputPtr1->getTensorDesc());
            input1->allocate();

            InferenceEngine::Blob::Ptr input2 = InferenceEngine::make_shared_blob<float>(inputPtr2->getTensorDesc());
            input2->allocate();

            InferenceEngine::BlobMap inputBlobs;
            inputBlobs["input1"] = input1;
            inputBlobs["input2"] = input2;

            OutputsDataMap outputs = network.getOutputsInfo();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(outputs["prior"]->getTensorDesc());
            output->allocate();

            InferenceEngine::BlobMap outputBlobs;
            outputBlobs["prior"] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(inputBlobs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            // Check results

            const TBlob<float>::Ptr outputArray = std::dynamic_pointer_cast<TBlob<float>>(output);
            float* dst_ptr = outputArray->data();

            // pick a few generated priors and compare against the expected number.
            // first prior
            EXPECT_NEAR(dst_ptr[0], 0.03, 1e-6);
            EXPECT_NEAR(dst_ptr[1], 0.03, 1e-6);
            EXPECT_NEAR(dst_ptr[2], 0.07, 1e-6);
            EXPECT_NEAR(dst_ptr[3], 0.07, 1e-6);
            // second prior
            EXPECT_NEAR(dst_ptr[4], 0.03, 0.1);
            EXPECT_NEAR(dst_ptr[5], 0.03, 0.1);
            EXPECT_NEAR(dst_ptr[6], 0.17, 0.1);
            EXPECT_NEAR(dst_ptr[7], 0.03, 0.1);
            // prior in the 5-th row and 5-th col
            EXPECT_NEAR(dst_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.83, 0.1);
            EXPECT_NEAR(dst_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.83, 0.1);
            EXPECT_NEAR(dst_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.84, 0.1);
            EXPECT_NEAR(dst_ptr[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.84, 0.1);

            // check variance
            dst_ptr += p.out.h * p.out.w;
            for (int d = 0; d < p.out.h * p.out.w; ++d) {
                EXPECT_NEAR(dst_ptr[d], 0.1, 1e-6);
            }
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(smoke_CPUPriorBoxDensityTest, TestsPriorBoxDensity) {}

INSTANTIATE_TEST_CASE_P(
    TestsPriorBoxDensity, smoke_CPUPriorBoxDensityTest,
    ::testing::Values(
        priorbox_test_params{ "CPU",
        10,{ 10, 10, 10 },{ 3, 100, 100 },{ 2, 1, 400 }, 0, 0, 4, 9, true, true }));

