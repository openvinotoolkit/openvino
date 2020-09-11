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


struct squeeze_unsqueeze_test_params {
    std::string device_name;
    std::string layerType;
    InferenceEngine::SizeVector in_dim;
    std::vector<int> squeeze_dims;
    InferenceEngine::SizeVector ref_dim;
    std::vector<float> ref;
};

template<typename data_t>
void ref_squeeze_unsqueeze(InferenceEngine::TBlob<float>& dst, squeeze_unsqueeze_test_params& prm) {
    data_t* dst_data = dst.buffer().template as<data_t*>();

    for (int i = 0; i < prm.ref.size(); ++i)
        dst_data[i] = prm.ref[i];
}

template<typename data_t>
InferenceEngine::TBlob<uint8_t>::Ptr generateWeights(const std::vector<int> &data) {
    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>(
        {InferenceEngine::Precision::U8,{ data.size() * sizeof(data_t) }, InferenceEngine::C}
    );
    weights->allocate();
    for (size_t i = 0; i < data.size(); i++) {
        ((data_t*) weights->buffer())[i] = data[i];
    }
    return InferenceEngine::TBlob<uint8_t>::Ptr(weights);
}

class SqueezeUnsqueezeTests : public TestsCommon, public WithParamInterface<squeeze_unsqueeze_test_params> {
    std::string model_t = R"V0G0N(
<net Name="squeeze_unsqueeze" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="Input1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer id="2" name="Input2" precision="FP32" type="Const">
            <output>
                <port id="0">
                    <dim>_INPUT_COUNT_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="4"/>
            </blobs>
        </layer>
        <layer name="squeeze_unsqueeze" id="5" type="_LAYER_" precision="FP32">
            <input>
                <port id="5">
                    _IN_
                </port>
                <port id="6">
                    <dim>_INPUT_COUNT_</dim>
                </port>
            </input>
            <output>
                <port id="9">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="5"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="6"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(squeeze_unsqueeze_test_params p) {
        std::string in, out;

        for (auto& i : p.in_dim) {
            in += "<dim>" + std::to_string(i) + "</dim>\n";
        }

        for (auto& o : p.ref_dim) {
            out += "<dim>" + std::to_string(o) + "</dim>\n";
        }

        REPLACE_WITH_STR(model_t, "_LAYER_", p.layerType);
        REPLACE_WITH_STR(model_t, "_IN_", in);
        REPLACE_WITH_STR(model_t, "_OUT_", out);
        REPLACE_WITH_NUM(model_t, "_INPUT_COUNT_", p.squeeze_dims.size());

        return model_t;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            squeeze_unsqueeze_test_params p = ::testing::WithParamInterface<squeeze_unsqueeze_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model,generateWeights<float>(p.squeeze_dims) );
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            InferenceEngine::OutputsDataMap out;
            out = net.getOutputsInfo();

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Input Data
            InferenceEngine::Blob::Ptr src;
            src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_dim, InferenceEngine::TensorDesc::getLayoutByDims(p.in_dim) });
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            ref_squeeze_unsqueeze<float>(dst_ref, p);

            inferRequest.SetBlob("Input1", src);
            inferRequest.Infer();

            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(SqueezeUnsqueezeTests, smoke_GPU_TestsSqueezeUnsqueeze) {}

//  Test data vectors
std::vector<float> squeeze_ref1 = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f };
std::vector<float> squeeze_ref2 = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
std::vector<float> squeeze_ref3 = { 0.f, 1.f, 2.f };

INSTANTIATE_TEST_CASE_P(
        smoke_TestsSqueezeUnsqueeze, SqueezeUnsqueezeTests,
        ::testing::Values(
                squeeze_unsqueeze_test_params{ "GPU", "Squeeze", { 1, 1, 3, 2 }, { 0, 1 }, { 3, 2, 1, 1 }, squeeze_ref1 },
                squeeze_unsqueeze_test_params{ "GPU", "Squeeze", { 3, 1, 3, 1 }, { 1 }, { 3, 3, 1, 1 }, squeeze_ref2 },
                squeeze_unsqueeze_test_params{ "GPU", "Squeeze", { 3, 1, 3, 1 }, { 3 }, { 3, 1, 3, 1 }, squeeze_ref2 },
                squeeze_unsqueeze_test_params{ "GPU", "Unsqueeze", { 3, 1, 1, 1 }, { 0, 2, 3 }, { 1, 3, 1, 1 }, squeeze_ref3 },
                squeeze_unsqueeze_test_params{ "GPU", "Unsqueeze", { 1, 1, 3, 1 }, { 0 }, { 1, 1, 1, 3 }, squeeze_ref3 },
                squeeze_unsqueeze_test_params{ "GPU", "Unsqueeze", { 1, 3, 1, 1 }, { 0, 1 }, { 1, 1, 1, 3 }, squeeze_ref3 },
                squeeze_unsqueeze_test_params{ "GPU", "Unsqueeze", { 3, 1, 1, 1 }, { 0, 1, 2 }, { 1, 1, 1, 3 }, squeeze_ref3 }
        ));
