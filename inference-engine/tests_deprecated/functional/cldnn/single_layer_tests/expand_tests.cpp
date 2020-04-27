// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"



using namespace ::testing;
using namespace InferenceEngine;
using namespace std;


struct broadcast_test_params {
    std::string device_name;
    InferenceEngine::SizeVector in_dim;
    InferenceEngine::SizeVector out_dim;
    std::vector<float> ref;
};

template<typename data_t>
void ref_broadcast(InferenceEngine::TBlob<data_t> &dsts, broadcast_test_params &prm) {
    data_t *dst_data = dsts.buffer().template as<data_t*>();
    for(int i = 0; i < prm.ref.size(); ++i)
        dst_data[i] = prm.ref[i];
}

InferenceEngine::TBlob<uint8_t>::Ptr generateWeights(const SizeVector &data) {
    TensorDesc td(InferenceEngine::Precision::U8, { data.size() * sizeof(uint32_t) }, InferenceEngine::C );
    TBlob<uint8_t>::Ptr weights;
    weights = make_shared_blob<uint8_t>(td);
    weights->allocate();
    auto wb = weights->buffer().as<uint32_t*>();
    for (size_t i = 0; i < data.size(); i++) {
        wb[i] = data[i];
    }
    return weights;
}


class BroadcastTests : public TestsCommon, public WithParamInterface<broadcast_test_params> {
    std::string model_t = R"V0G0N(
<net Name="broadcast" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="input2" type="Const" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>4</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="4"/>
            </blobs>
        </layer>
        <layer name="broadcast" id="2" type="Broadcast" precision="FP32">
            <input>
                <port id="0">
                    _IN_
                </port>
                <port id="1">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(broadcast_test_params p) {
        std::string in, out;

        for (auto& i : p.in_dim) {
            in += "<dim>" + std::to_string(i) + "</dim>\n";
        }

        for (auto& o : p.out_dim) {
            out += "<dim>" + std::to_string(o) + "</dim>\n";
        }

        REPLACE_WITH_STR(model_t, "_IN_", in);
        REPLACE_WITH_STR(model_t, "_OUT_", out);

        return model_t;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            broadcast_test_params p = ::testing::WithParamInterface<broadcast_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, generateWeights(p.out_dim));
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Input Data
            InputsDataMap inputInfo(net.getInputsInfo());
            Blob::Ptr inputBlob = inferRequest.GetBlob(inputInfo.begin()->first);
            float* inputData = inputBlob->buffer().as<float*>();
            fill_data_dbgval(inputData, inputBlob->size());

            inferRequest.Infer();

            // Output Data
            OutputsDataMap outputInfo(net.getOutputsInfo());
            Blob::Ptr outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);

            size_t outSz = outputBlob->size();
            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(outputBlob->getTensorDesc());
            dst_ref.allocate();
            ref_broadcast<float>(dst_ref, p);

            const float* res = outputBlob->buffer().as<float*>();
            const float* ref = dst_ref.data();
            compare(res, ref, outSz);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(BroadcastTests, smoke_GPU_TestsBroadcast) {}

//  Test data vectors
std::vector<float> broadcast_ref0 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
std::vector<float> broadcast_ref1 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f};
std::vector<float> broadcast_ref2 = { 0.f, 0.f, 0.f, 0.f,
                                      1.f, 1.f, 1.f, 1.f,
                                      2.f, 2.f, 2.f, 2.f};

INSTANTIATE_TEST_CASE_P(
        smoke_TestsBroadcast, BroadcastTests,
        ::testing::Values(
                broadcast_test_params{ "GPU", { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, broadcast_ref0 },
                broadcast_test_params{ "GPU", { 1, 1, 3, 1 }, { 1, 2, 3, 6 }, broadcast_ref1 },
                broadcast_test_params{ "GPU", { 1, 1, 3, 1 }, { 1, 1, 3, 4 }, broadcast_ref2 }
        ));
