// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "xml_net_builder.hpp"
#include "tests_common.hpp"
#include "precision_utils.h"

#include "functional_test_utils/plugin_cache.hpp"

#include <iostream>
#include <string>
#include <map>

using namespace InferenceEngine;
using namespace CommonTestUtils;

struct conv_param {
    size_t k_x, k_y;
    size_t p_x, p_y;
    size_t s_x, s_y;

    std::vector<size_t> in;
    std::vector<size_t> out;
};

using test_param = std::tuple<
    std::string,                      // Plugin name
    std::tuple<
        conv_param,                   // Convolution params
        std::pair<Precision, float>,  // Network precision
        Layout, Layout,               // Layout in/out
        Precision                     // Precision input
    >
>;

class LayoutTTTest : public ::testing::TestWithParam<test_param> {
protected:
    conv_param cnv;
    Precision netPrc;
    float threshold;
    std::string _device;
    std::shared_ptr<InferenceEngine::Core> ie;
    std::map<std::string, std::string> _deviceConfig;
    Layout l_in, l_out;
    Precision prc_in;

    std::string model;
    TBlob<uint8_t>::Ptr weights;
    test_param p;

    size_t N, IC, IH, IW, OC, OH, OW;

    void SetUp() override {
        _device = std::get<0>(GetParam());
        ie = PluginCache::get().ie();
        std::pair<Precision, float> netPrcThresh;
        std::tie(
            cnv,
            netPrcThresh,
            l_in, l_out,
            prc_in
        ) = std::get<1>(GetParam());

        /*======== Some additional params =========*/
        N  = cnv.in[0];
        IC = cnv.in[1];
        IH = cnv.in[2];
        IW = cnv.in[3];
        OC = cnv.out[1];
        OH = cnv.out[2];
        OW = cnv.out[3];

        netPrc = netPrcThresh.first;
        threshold = netPrcThresh.second;

        if (netPrc == Precision::FP32) {
            prepareNetwork<Precision::FP32>();
        } else {
            prepareNetwork<Precision::FP16>();
        }

        if (_device == "HETERO")
            _deviceConfig["TARGET_FALLBACK"] = "GPU,CPU";
    }

    template <Precision::ePrecision PRC>
    void prepareNetwork() {
        using data_t = typename PrecisionTrait<PRC>::value_type;

        /*======== Prepare model IR =========*/
        std::map<std::string, std::string> conv_params = {
                { "stride-x", std::to_string(cnv.s_x) },
                { "stride-y", std::to_string(cnv.s_y) },
                { "pad-x",    std::to_string(cnv.p_x) },
                { "pad-y",    std::to_string(cnv.p_y) },
                { "kernel-x", std::to_string(cnv.k_x) },
                { "kernel-y", std::to_string(cnv.k_y) },
                { "output",   std::to_string(cnv.out[1])},
                { "group",    "1" }
        };

        InOutShapes inout = { {cnv.in}, {cnv.out} };

        size_t KH = cnv.k_y;
        size_t KW = cnv.k_x;
        std::ostringstream prc_name;
        prc_name << PRC;

        model = V2NetBuilder::buildNetworkWithOneInput("ConvNet", cnv.in, prc_name.str())
                .addLayer("Convolution", prc_name.str(), &conv_params, inout, OC*IC*KH*KW*sizeof(data_t), OC*sizeof(data_t))
                .finish(false);

        /*======== Prepare model Weights =========*/
        weights = make_shared_blob<uint8_t>(
            {Precision::U8, { ( OC*IC*KH*KW + OC )*sizeof(data_t) }, Layout::C});
        weights->allocate();

        data_t* w_ptr = weights->buffer().as<data_t*>();

        for (size_t oc = 1; oc < OC+1; oc++)
            for (size_t ic = 1; ic < IC+1; ic++)
                for (size_t khw = 0; khw < KH*KW; khw++) {
                    if (PRC == Precision::FP32) {
                        *w_ptr++ = 1.0 / (ic * KH * KW * IC) * oc;
                    } else {
                        *w_ptr++ = PrecisionUtils::f32tof16(1.0 / (ic * KH * KW * IC) * oc);
                    }
                }

        for (size_t oc = 1; oc < OC+1; oc++) {
            if (PRC == Precision::FP32) {
                *w_ptr++ = oc;
            } else {
                *w_ptr++ = PrecisionUtils::f32tof16(oc);
            }
        }
    }

    void TearDown() override {
        PluginCache::get().reset();
    }

    template <Precision::ePrecision PRC>
    Blob::Ptr input() {
        using data_t = typename PrecisionTrait<PRC>::value_type;

        SizeVector in_dims(cnv.in.begin(), cnv.in.end());
        Blob::Ptr input = make_shared_blob<data_t>({PRC, in_dims, l_in});
        input->allocate();
        data_t* i_ptr = input->buffer().as<data_t*>();

        if (l_in == NCHW) {
            for (size_t n = 0; n < N; n++)
            for (size_t c = 1; c < IC + 1; c++)
            for (size_t hw = 0; hw < IH * IW; hw++)
                *i_ptr++ = c;
        } else { // NCHW
            for (size_t n = 0; n < N; n++)
            for (size_t hw = 0; hw < IH * IW; hw++)
            for (size_t c = 1; c < IC + 1; c++)
                *i_ptr++ = c;
        }

        return input;
    }

    Blob::Ptr input(Precision prc) {
        return prc == Precision::FP32 ? input<Precision::FP32>():
               prc == Precision::U8   ? input<Precision::U8>()  :
               prc == Precision::I8   ? input<Precision::I8>() :
               prc == Precision::I16  ? input<Precision::I16>() :
               prc == Precision::U16  ? input<Precision::U16>() :
                                        input<Precision::FP32>();
    }

    void checkOutput(Blob::Ptr output) {
        float *o_ptr = output->buffer().as<float*>();

        if (l_out == NCHW) {
            for (size_t n = 0; n < N; n++)
            for (size_t c = 1; c < OC+1; c++)
            for (size_t hw = 0; hw < OH*OW; hw++)
                ASSERT_NEAR(*o_ptr++, c*2, threshold);
        } else {
            for (size_t n = 0; n < N; n++)
            for (size_t hw = 0; hw < OH*OW; hw++)
            for (size_t c = 1; c < OC+1; c++)
                ASSERT_NEAR(*o_ptr++, c*2, threshold);
        };
    }

};

TEST_P(LayoutTTTest, SomeTest1) {
    CNNNetwork net = ie->ReadNetwork(model, weights);

    net.getInputsInfo().begin()->second->setPrecision(prc_in);
    net.getInputsInfo().begin()->second->setLayout(l_in);
    net.getOutputsInfo().begin()->second->setLayout(l_out);

    auto execNet = ie->LoadNetwork(net, _device, _deviceConfig);
    auto req = execNet.CreateInferRequest();

    req.SetBlob("Input0", input(prc_in));
    req.Infer();

    Blob::Ptr output = req.GetBlob("Convolution1");
    checkOutput(output);
}

conv_param conv_p = { 3,3, // kernel
                      0,0, // pads
                      1,1, // strides
                      {2,3,15,15},   // in shape
                      {2,16,13,13}}; // out shape

#define PLUGING_CASE(_device, _test, _params) \
    INSTANTIATE_TEST_CASE_P(_device##_run, _test, ::testing::Combine(::testing::Values(#_device), _params) )

#define PLUGING_CASE_WITH_SUFFIX(_device, _suffix, _test, _params) \
    INSTANTIATE_TEST_CASE_P(_device##_run##_suffix, _test, ::testing::Combine(::testing::Values(#_device), _params) )

#define PLUGING_CASE_WITH_PREFIX(_device, _prefix, _test, _params) \
    INSTANTIATE_TEST_CASE_P(_prefix##_device##_run, _test, ::testing::Combine(::testing::Values(#_device), _params) )
