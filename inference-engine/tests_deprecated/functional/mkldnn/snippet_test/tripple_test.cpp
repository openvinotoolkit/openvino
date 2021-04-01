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

const static SizeVector dims {_B, _C, _H, _W};

class TripleConnectNet : CommonTestUtils::V2NetBuilder {
    std::string model;
    TBlob<uint8_t>::Ptr weightsPtr;

public:
	TripleConnectNet(): CommonTestUtils::V2NetBuilder(buildNetworkWithOneInput(
            "Triple_Net", {_B, _C, _H, _W}, "FP32")) {
		weightsPtr = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, SizeVector{0}, Layout::C));
		weightsPtr->allocate();

		/**
		 *      [in]
		 *     ___|___
		 *    |   |   |
         *   [0] [1] [2]
		 *  [__Concat___]
		 *        |
		 *      [out]
		 */
        map<string, string> lstm_params = {};
        addLayer("Concat", "FP32",
                 &lstm_params,
                 {  // input dims
					{dims, dims, dims},
					// output dims
					{{_B, 3*_C, _H, _W}}
				 });

        vector<pair<string, string>> edges = {
                {"0,0", "1,1"},
                {"0,0", "1,2"},
                {"0,0", "1,3"}
        };
        model = finish(&edges);
    }

    CNNNetwork net(Core & ie) {
        return ie.ReadNetwork(model, weightsPtr);
    }
};

using test_param = std::tuple<string>;

class smoke_TripleConnectTest : public ::testing::TestWithParam<test_param> {
protected:
    string device_name;
    TripleConnectNet topology;

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

TEST_P(smoke_TripleConnectTest, canLoad) {
    Core ie;
    CNNNetwork net = topology.net(ie);

    auto execNet = ie.LoadNetwork(net, device_name);
    auto req = execNet.CreateInferRequest();

    auto input = req.GetBlob("Input0");
    fill_with(input, {1,2,3,4});

    req.Infer();

    auto output = req.GetBlob("Concat1");
    ASSERT_TRUE(check_with(output, {1,2,3,4}));
}

#define PLUGING_CASE(_plugin, _test) \
    INSTANTIATE_TEST_CASE_P(_plugin##_run, _test, ::testing::Values(#_plugin) )

PLUGING_CASE(CPU, smoke_TripleConnectTest);
