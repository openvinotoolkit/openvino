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

struct padTF_test_params {
    std::string device;
    SizeVector in_size;
    std::vector<float> in;
    SizeVector pads_begin;
    SizeVector pads_end;
    std::string pad_mode;
    float pad_value;
    SizeVector ref_size;
    std::vector<float> ref;
};

class PadTFTests : public TestsCommon, public WithParamInterface<padTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Pad_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Pad" precision="FP32">
            <data pads_begin="_P_BEGIN_" pads_end="_P_END_" pad_mode="_P_MODE_" pad_value="_P_VAL_"/>
            <input>
                <port id="2">
                    _IN_
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
        <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(padTF_test_params p) {
        std::string model = model_t;
        std::string in_size;
        std::string pads_begin;
        std::string pads_end;
        std::string ref_size;

        for (auto& src : p.in_size) {
            in_size += "<dim>";
            in_size += std::to_string(src) + "</dim>\n";
        }

        for (auto& pb : p.pads_begin)
            pads_begin += std::to_string(pb) + ",";
        pads_begin.pop_back();

        for (auto& pe : p.pads_end)
            pads_end += std::to_string(pe) + ",";
        pads_end.pop_back();

        for (auto& dst : p.ref_size) {
            ref_size += "<dim>";
            ref_size += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IN_", in_size);
        REPLACE_WITH_STR(model, "_P_BEGIN_", pads_begin);
        REPLACE_WITH_STR(model, "_P_END_", pads_end);
        REPLACE_WITH_STR(model, "_P_MODE_", p.pad_mode);
        REPLACE_WITH_NUM(model, "_P_VAL_", p.pad_value);
        REPLACE_WITH_STR(model, "_OUT_", ref_size);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            padTF_test_params p = ::testing::WithParamInterface<padTF_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            InferenceEngine::Blob::Ptr src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_size, InferenceEngine::TensorDesc::getLayoutByDims(p.in_size) });
            src->allocate();
            float* psrc = src->buffer().as<float*>();
            std::copy(p.in.begin(), p.in.end(), psrc);
            inferRequest.SetBlob("input", src);

            InferenceEngine::Blob::Ptr dst = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.ref_size, InferenceEngine::TensorDesc::getLayoutByDims(p.ref_size) });
            dst->allocate();
            inferRequest.SetBlob("output", dst);

            //  Infer
            inferRequest.Infer();

            //  Check results
            TBlob<float> dst_ref({ Precision::FP32, p.ref_size, TensorDesc::getLayoutByDims(p.ref_size) });
            dst_ref.allocate();
            float* pdst_ref = dst_ref.buffer().as<float*>();
            std::copy(p.ref.begin(), p.ref.end(), pdst_ref);
            compare(*dst, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(PadTFTests, TestsPad) {}

std::string getTestName(testing::TestParamInfo<padTF_test_params> obj) {
    std::string name = obj.param.device + "_" + obj.param.pad_mode;
    return name;
}

//  Examples of the standalone Pad operation input / output:
std::vector<float> in =
{1.f, 2.f, 3.f, 4.f,
 5.f, 6.f, 7.f, 8.f,
 9.f,10.f,11.f,12.f}; //  3x4

std::vector<float> ref_constant =
{0.f,0.f,0.f, 0.f, 0.f, 0.f,0.f,0.f,0.f,
 0.f,0.f,0.f, 0.f, 0.f, 0.f,0.f,0.f,0.f,
 0.f,0.f,1.f, 2.f, 3.f, 4.f,0.f,0.f,0.f,
 0.f,0.f,5.f, 6.f, 7.f, 8.f,0.f,0.f,0.f,
 0.f,0.f,9.f,10.f,11.f,12.f,0.f,0.f,0.f,
 0.f,0.f,0.f, 0.f, 0.f, 0.f,0.f,0.f,0.f}; //  6x9

std::vector<float> ref_edge =
{1.f,1.f,1.f, 2.f, 3.f, 4.f, 4.f, 4.f, 4.f,
 1.f,1.f,1.f, 2.f, 3.f, 4.f, 4.f, 4.f, 4.f,
 1.f,1.f,1.f, 2.f, 3.f, 4.f, 4.f, 4.f, 4.f,
 5.f,5.f,5.f, 6.f, 7.f, 8.f, 8.f, 8.f, 8.f,
 9.f,9.f,9.f,10.f,11.f,12.f,12.f,12.f,12.f,
 9.f,9.f,9.f,10.f,11.f,12.f,12.f,12.f,12.f}; //  6x9

std::vector<float> ref_reflect =
{11.f,10.f,9.f,10.f,11.f,12.f,11.f,10.f,9.f,
  7.f, 6.f,5.f, 6.f, 7.f, 8.f, 7.f, 6.f,5.f,
  3.f, 2.f,1.f, 2.f, 3.f, 4.f, 3.f, 2.f,1.f,
  7.f, 6.f,5.f, 6.f, 7.f, 8.f, 7.f, 6.f,5.f,
 11.f,10.f,9.f,10.f,11.f,12.f,11.f,10.f,9.f,
  7.f, 6.f,5.f, 6.f, 7.f, 8.f, 7.f, 6.f,5.f}; //  6x9

std::vector<float> ref_symmetric =
{6.f,5.f,5.f, 6.f, 7.f, 8.f, 8.f, 7.f, 6.f,
 2.f,1.f,1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f,
 2.f,1.f,1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f,
 6.f,5.f,5.f, 6.f, 7.f, 8.f, 8.f, 7.f, 6.f,
10.f,9.f,9.f,10.f,11.f,12.f,12.f,11.f,10.f,
10.f,9.f,9.f,10.f,11.f,12.f,12.f,11.f,10.f}; //  6x9

#define PLUGING_CASE(_device, _test, __num, ...) \
    INSTANTIATE_TEST_CASE_P(smoke_##_device##_run##__num, _test, ::testing::Values(padTF_test_params{#_device, __VA_ARGS__}) );

#define PLUGING_CASE_WITH_SUFFIX(_device, _suffix, _test, __num, ...) \
    INSTANTIATE_TEST_CASE_P(_device##_run##_suffix##__num, _test, ::testing::Values(padTF_test_params{#_device, __VA_ARGS__}) );
