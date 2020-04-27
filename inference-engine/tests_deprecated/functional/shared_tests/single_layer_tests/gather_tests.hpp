// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "ie_memcpy.h"

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;


struct gatherTF_test_params {
    std::string device_name;

    std::string inIdxPrecision;

    SizeVector in_dim;
    std::vector<float> in;

    SizeVector dct_dim;
    std::vector<float> dct;

    int axis;

    SizeVector ref_dim;
    std::vector<float> ref;
};

class GatherTFTests : public TestsCommon, public WithParamInterface<gatherTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Gather_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputDictionary" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IDICT_
                </port>
            </output>
        </layer>
        <layer name="InputText" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    _IIDX_
                </port>
            </output>
        </layer>
        <layer name="gather" id="3" type="Gather" precision="FP32">
            <data axis="_AX_"/>
            <input>
                <port id="1">
                    _IDICT_
                </port>
                <port id="2">
                    _IIDX_
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(gatherTF_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inDict;
        std::string out;

        for (auto& idx : p.in_dim) {
            inIdx += "<dim>";
            inIdx += std::to_string(idx) + "</dim>\n";
        }

        for (auto& dct : p.dct_dim) {
            inDict += "<dim>";
            inDict += std::to_string(dct) + "</dim>\n";
        }

        for (auto& dst : p.ref_dim) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_STR(model, "_IIDX_", inIdx);
        REPLACE_WITH_STR(model, "_IDICT_", inDict);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            gatherTF_test_params p = ::testing::WithParamInterface<gatherTF_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Input Indexes
            Blob::Ptr srcIdx;
            if (p.inIdxPrecision == "I32") {
                srcIdx = make_shared_blob<int32_t>({Precision::I32, p.in_dim,
                                                                   TensorDesc::getLayoutByDims(
                                                                           p.in_dim)});
                srcIdx->allocate();
                auto *srcIdxPtr = dynamic_cast<TBlob<int32_t> *>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                int32_t *srcIdxP = (int32_t*)srcIdx->buffer();
                for (int i=0; i<p.in.size(); i++)
                    srcIdxP[i] = static_cast<int32_t>(p.in[i]);
            } else if (p.inIdxPrecision == "FP32") {
                srcIdx = make_shared_blob<float>({Precision::FP32, p.in_dim,
                                                                    TensorDesc::getLayoutByDims(
                                                                            p.in_dim)});
                srcIdx->allocate();
                auto *srcIdxPtr = dynamic_cast<TBlob<float> *>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
                ie_memcpy(static_cast<float *>(srcIdx->buffer()), srcIdx->byteSize(), &p.in[0], sizeof(float) * p.in.size());
            } else if (p.inIdxPrecision == "I8") {
                srcIdx = make_shared_blob<int8_t>({Precision::I8, p.in_dim,
                                                                   TensorDesc::getLayoutByDims(
                                                                           p.in_dim)});
                srcIdx->allocate();
                auto *srcIdxPtr = dynamic_cast<TBlob<int8_t> *>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
                int8_t *srcIdxP = (int8_t*)srcIdx->buffer();
                for (int i=0; i<p.in.size(); i++)
                    srcIdxP[i] = static_cast<int8_t>(p.in[i]);
            } else if (p.inIdxPrecision == "I16") {
                srcIdx = make_shared_blob<int16_t>({Precision::I16, p.in_dim,
                                                                    TensorDesc::getLayoutByDims(
                                                                            p.in_dim)});
                srcIdx->allocate();
                auto *srcIdxPtr = dynamic_cast<TBlob<int16_t> *>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int16_t>.";
                int16_t *srcIdxP = (int16_t*)srcIdx->buffer();
                for (int i=0; i<p.in.size(); i++)
                    srcIdxP[i] = static_cast<int16_t>(p.in[i]);
            }

            //  Input Dictionary
            Blob::Ptr srcDict = make_shared_blob<float>({ Precision::FP32,
                                                        p.dct_dim, TensorDesc::getLayoutByDims(p.dct_dim) });
            srcDict->allocate();
            ie_memcpy(srcDict->buffer(), srcDict->byteSize(), &p.dct[0], sizeof(float)*p.dct.size());
            auto * srcDictPtr = dynamic_cast<TBlob<float>*>(srcDict.get());
            if (srcDictPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            //  Output Data
            OutputsDataMap out = net.getOutputsInfo();
            std::pair<std::string, DataPtr> item = *out.begin();
            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);

            //  Infer
            inferRequest.SetBlob("InputDictionary", srcDict);
            inferRequest.SetBlob("InputText", srcIdx);
            inferRequest.Infer();

            //  Check results
            if (memcmp((*output).data(), &p.ref[0], p.ref.size() * sizeof(float)) != 0)
                FAIL() << "Wrong result with compare TF reference!";
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(GatherTFTests, TestsGather) {}

//  Test data vectors
std::vector<float> in0 = { 0.f, 1.f, 1.f, 0.f };
std::vector<float> in1 = { 0.f, 1.f, 2.f, 1.f };
std::vector<float> dict = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f };
std::vector<float> dict2D = { 1.f, 2.f, 3.f, 4.f}; // 2x2
std::vector<float> ref_in0_a0_d223 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }; // 2x2x2x3
std::vector<float> ref_in0_a2_d232 = { 1.f, 2.f, 2.f, 1.f, 3.f, 4.f, 4.f, 3.f, 5.f, 6.f, 6.f, 5.f, 7.f, 8.f, 8.f, 7.f, 9.f, 10.f, 10.f, 9.f, 11.f, 12.f, 12.f, 11.f }; // 2x3x2x2
std::vector<float> ref_in1_a0_d322 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 5.f, 6.f, 7.f, 8.f }; // 2x2x2x2
std::vector<float> ref_in1_a1_d232 = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 3.f, 4.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f }; // 2x2x2x2
std::vector<float> ref_in1_a2_d223 = { 1.f, 2.f, 3.f, 2.f, 4.f, 5.f, 6.f, 5.f, 7.f, 8.f, 9.f, 8.f, 10.f, 11.f, 12.f, 11.f }; // 2x2x2x2
std::vector<float> ref_in0_a0_d22 = { 1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 1.f, 2.f }; // 2x2x2
