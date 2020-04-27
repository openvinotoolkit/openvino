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


struct shuffle_channels_test_params {
    std::string device_name;
    std::string inPrecision;
    SizeVector in_out_shape;
    int axis;
    int group;
    std::vector<float> reference;
};

void ref_shuffle_channels(TBlob<float> &src, TBlob<float> &dst, int axis, int group) {
    size_t i;
    const float *src_data = src.data();
    float* dst_data = dst.data();
    SizeVector dst_dims = dst.getTensorDesc().getDims();
    SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();

    if (axis < 0)
        axis += dst_dims.size();

    if (axis < 0 || axis >= dst_dims.size())
        FAIL() << "Incorrect input parameters dimensions and axis number!";

    if (dst_dims[axis] % group)
        FAIL() << "Group parameter must evenly divide the channel dimension!";

    //  Find number of dictionaries, index range and data length
    size_t numDictionaries = 1;
    for (i = 0; i <= axis; i++)
        numDictionaries *= dst_dims[i];

    size_t channelsNum = dst_dims[axis] / group;

    size_t dataLength = 1;
    for (i = axis + 1; i < dst_dims.size(); i++)
        dataLength *= dst_dims[i];

    if (dataLength == 0)
        FAIL() << "Incorrect input parameters dimension!";

    size_t j, k;
    for (j = 0, k = 0; j < numDictionaries; j += dst_dims[axis]) {
        for (i = 0; i < (dst_dims[axis] * channelsNum); i += channelsNum, k += dataLength) {
            int idx = j + i / dst_dims[axis] + i % dst_dims[axis];
            memcpy(&dst_data[k], &src_data[dataLength * idx], sizeof(float) * dataLength);
        }
    }
}

class ShuffleChannelsTests : public TestsCommon, public WithParamInterface<shuffle_channels_test_params> {
    std::string model_t = R"V0G0N(
<net Name="ShuffleChannels_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_OUT_
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="ShuffleChannels" precision="FP32">
            <data axis="_AX_" group="_GR_"/>
            <input>
                <port id="1">
                    _IN_OUT_
                </port>
           </input>
            <output>
                <port id="2">
                    _IN_OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(shuffle_channels_test_params p) {
        std::string model = model_t;
        std::string in_out_shape;

        for (size_t i = 0; i < p.in_out_shape.size(); i++) {
            in_out_shape += "<dim>";
            in_out_shape += std::to_string(p.in_out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_OUT_", in_out_shape);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);
        REPLACE_WITH_NUM(model, "_GR_", p.group);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            shuffle_channels_test_params p = ::testing::WithParamInterface<shuffle_channels_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());

            // Output Data
            OutputsDataMap out = net.getOutputsInfo();
            auto item = *out.begin();

            // Input Data
            Blob::Ptr src = make_shared_blob<float>({Precision::FP32,
                p.in_out_shape,
                TensorDesc::getLayoutByDims(p.in_out_shape)});
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());
            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast input blob to TBlob<float>.";

            // Output Reference
            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_shuffle_channels(*srcPtr, dst_ref, p.axis, p.group);

            // Check results
            if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                FAIL() << "Wrong result of TF reference comparison!";

            // Infer
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();
            inferRequest.SetBlob("input", src);
            auto output = inferRequest.GetBlob(item.first);

            inferRequest.Infer();

            // Check results
            auto * dstPtr = dynamic_cast<TBlob<float>*>(output.get());
            if (dstPtr == nullptr)
                FAIL() << "Cannot cast output blob to TBlob<float>.";
            compare(*dstPtr, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(ShuffleChannelsTests, smoke_GPU_TestsShuffleChannels) {}

static std::vector<float> test0 = { 0.f, 1.f, 2.f, 3.f, 12.f, 13.f, 14.f, 15.f, 24.f, 25.f, 26.f, 27.f, 36.f, 37.f, 38.f, 39.f, 48.f, 49.f, 50.f, 51.f,
                                    4.f, 5.f, 6.f, 7.f, 16.f, 17.f, 18.f, 19.f, 28.f, 29.f, 30.f, 31.f, 40.f, 41.f, 42.f, 43.f, 52.f, 53.f, 54.f, 55.f,
                                    8.f, 9.f, 10.f, 11.f, 20.f, 21.f, 22.f, 23.f, 32.f, 33.f, 34.f, 35.f, 44.f, 45.f, 46.f, 47.f, 56.f, 57.f, 58.f, 59.f };
static std::vector<float> test4 = { 0.f, 2.f, 4.f, 1.f, 3.f, 5.f, 6.f, 8.f, 10.f, 7.f, 9.f, 11.f, 12.f, 14.f, 16.f, 13.f, 15.f, 17.f, 18.f, 20.f, 22.f, 19.f, 21.f, 23.f };
static std::vector<float> test5 = { 0.f, 1.f, 4.f, 5.f, 8.f, 9.f, 2.f, 3.f, 6.f, 7.f, 10.f, 11.f, 12.f, 13.f, 16.f, 17.f, 20.f, 21.f, 14.f, 15.f, 18.f, 19.f, 22.f, 23.f };
static std::vector<float> test6 = { 0.f, 3.f, 1.f, 4.f, 2.f, 5.f, 6.f, 9.f, 7.f, 10.f, 8.f, 11.f, 12.f, 15.f, 13.f, 16.f, 14.f, 17.f, 18.f, 21.f, 19.f, 22.f, 20.f, 23.f };
static std::vector<float> test7 = { 0.f, 1.f, 6.f, 7.f, 2.f, 3.f, 8.f, 9.f, 4.f, 5.f, 10.f, 11.f, 12.f, 13.f, 18.f, 19.f, 14.f, 15.f, 20.f, 21.f, 16.f, 17.f, 22.f, 23.f };
static std::vector<float> test8 = { 0.f, 3.f, 1.f, 4.f, 2.f, 5.f };

INSTANTIATE_TEST_CASE_P(
        smoke_TestsShuffleChannels, ShuffleChannelsTests,
        ::testing::Values(
                shuffle_channels_test_params{ "GPU", "FP32", { 1, 15, 2, 2 }, 1, 5, test0 },
                shuffle_channels_test_params{ "GPU", "FP32", { 1, 15, 2, 2 }, -3, 5, test0 },
                shuffle_channels_test_params{ "GPU", "FP32", { 15, 2, 2 }, 0, 5, test0 },
                shuffle_channels_test_params{ "GPU", "FP32", { 15, 2, 2 }, -3, 5, test0 },
                shuffle_channels_test_params{ "GPU", "FP32", { 2, 2, 6 }, -1, 3, test4 },
                shuffle_channels_test_params{ "GPU", "FP32", { 2, 6, 2 }, -2, 3, test5 },
                shuffle_channels_test_params{ "GPU", "FP32", { 2, 2, 6 }, -1, 2, test6 },
                shuffle_channels_test_params{ "GPU", "FP32", { 2, 6, 2 }, -2, 2, test7 }
        ));

