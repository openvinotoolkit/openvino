// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/xml_net_builder/xml_net_builder.hpp"
#include "tests_common.hpp"
#include "precision_utils.h"
#include <ie_core.hpp>

using namespace InferenceEngine;
using std::string;
using std::pair;
using std::map;
using std::vector;

const static size_t _H = 16;
const static size_t _W = 16;
const static size_t _C = 1;
const static size_t _B = 2;

const static SizeVector dims    {_B, _C, _H, _W};

class MultiOutConnectNet : CommonTestUtils::V2NetBuilder {
    std::string model;
    TBlob<uint8_t>::Ptr weightsPtr;

public:
    MultiOutConnectNet(): CommonTestUtils::V2NetBuilder(buildNetworkWithOneInput(
            "MultiOutNet", {_B, 3*_C, _H, _W}, "FP32")) {
		weightsPtr = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, SizeVector{0}, Layout::C));
		weightsPtr->allocate();

		/**
		 *      [in]
		 *        |
		 *   [__split__]
		 *    |   |   |
         * [out1] |  [out2]
         *        |_______
         *        |       |
		 *   [power1]   [power2]
		 *        |       |
		 *     [out3]   [out4]
		 */
        addLayer("Split", "FP32", nullptr,
                 { {{_B, 3*_C, _H, _W}},
                   {dims, dims, dims}});

        map<string, string> pow_params = { {"scale", "-1"}, {"shift", "0"}, {"power", "1"} };
        addLayer("Power", "FP32", &pow_params,
                 { {dims}, {dims} });

        addLayer("Power", "FP32", &pow_params,
                 { {dims}, {dims} });

        vector<pair<string, string>> edges = {
                {"0,0", "1,1"},
                {"1,3", "2,5"},
                {"1,3", "3,7"}
        };
        model = finish(&edges);
    }

    CNNNetwork net(Core & ie) {
        return ie.ReadNetwork(model, weightsPtr);
    }
};

using test_param = std::tuple<string>;

class smoke_MultiOutConnectTest : public ::testing::TestWithParam<test_param> {
protected:
    string device_name;
    MultiOutConnectNet topology;

    void SetUp() override {
        device_name = std::get<0>(GetParam());
    }
};

static void fill_with(Blob::Ptr &blob, std::vector<float> vals) {
    float* ptr = blob->buffer().as<float*>();
    const size_t size = blob->size();
    const size_t fill_size = vals.size();

    for (int i = 0; i < size; i++)
        ptr[i] = vals[i%fill_size];
}

static bool check_with(Blob::Ptr &blob, std::vector<float> vals) {
    float* ptr = blob->buffer().as<float*>();
    const size_t size = blob->size();
    const size_t fill_size = vals.size();

    bool res = true;
    for (int i = 0; i < size; i++)
        if (ptr[i] != vals[i%fill_size])
            res = false;
    return res;
}

TEST_P(smoke_MultiOutConnectTest, canLoad) {
    Core ie;
    CNNNetwork net = topology.net(ie);

    auto execNet = ie.LoadNetwork(net, device_name);
    auto req = execNet.CreateInferRequest();

    auto input = req.GetBlob("Input0");
    fill_with(input, {1,2,3,4});

    req.Infer();

    auto output1 = req.GetBlob("Power2");
    auto output2 = req.GetBlob("Power3");
    ASSERT_TRUE(check_with(output1, {-1,-2,-3,-4}));
    ASSERT_TRUE(check_with(output2, {-1,-2,-3,-4}));
}

#define PLUGING_CASE(_plugin, _test) \
    INSTANTIATE_TEST_CASE_P(_plugin##_run, _test, ::testing::Values(#_plugin) )

PLUGING_CASE(CPU, smoke_MultiOutConnectTest);
