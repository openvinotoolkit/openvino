// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <memory>
#include "xml_net_builder.hpp"
#include "tests_common.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include <string>
#include <map>

using namespace InferenceEngine;

using test_param = std::tuple<
    std::string,   // Plugin name
    std::tuple<
        Precision  // Network Precision
    >
>;

class IO_BlobTest : public ::testing::TestWithParam<test_param> {
protected:
    std::string deviceName;
    std::shared_ptr<InferenceEngine::Core> ie;
    std::map<std::string, std::string> deviceConfig;
    Precision netPrc;
    CNNNetwork net;

    void SetUp() override {
        // default plugin instantiation
        deviceName = std::get<0>(GetParam());
        std::tie(
                netPrc
        ) = std::get<1>(GetParam());

        // default model: in->con->out
        auto weights = make_shared_blob<uint8_t>({Precision::U8, {0}, Layout::C});
        std::string model = ConvNet(/*MB*/ 2, weights);

        ie = PluginCache::get().ie();

        // loaded network
        net = ie->ReadNetwork(model, weights);

        if (deviceName == "HETERO")
            deviceConfig["TARGET_FALLBACK"] = "GPU,CPU";
    }

    void TearDown() override {
        PluginCache::get().reset();
    }

    std::string ConvNet(const int batch, TBlob<uint8_t>::Ptr &weights) {
        if (netPrc == Precision::FP32) {
            return ConvNetImpl<Precision::FP32>(batch, weights);
        } else {
            return ConvNetImpl<Precision::FP16>(batch, weights);
        }
    }

    template <Precision::ePrecision PRC>
    std::string ConvNetImpl(const int batch, TBlob<uint8_t>::Ptr &weights) {
        using data_t = typename PrecisionTrait<PRC>::value_type;


        size_t IC = 3, OC = 3, KH = 3, KW = 3;
        std::vector<size_t> in{static_cast<size_t>(batch), IC, 25, 25};
        std::vector<size_t> out{static_cast<size_t>(batch), OC, 25, 25};

        std::map<std::string, std::string> params{
#define PAR(_key, _val) { _key, std::to_string(_val) }
                PAR("stride-x", 1),
                PAR("stride-y", 1),
                PAR("pad-x", 1),
                PAR("pad-y", 1),
                PAR("kernel-x", 3),
                PAR("kernel-y", 3),
                PAR("output", 3),
                PAR("group", 1),
#undef  PAR
        };

        std::ostringstream prc_name;
        prc_name << PRC;

        weights = make_shared_blob<uint8_t>({weights->getTensorDesc().getPrecision(),
                                            { (OC * IC * KH * KW + OC) * sizeof(data_t) },
                                            weights->getTensorDesc().getLayout()});
        weights->allocate();
        TestsCommon::fill_data(weights->buffer().as<float*>(),
                  weights->size() / sizeof(float));
        return CommonTestUtils::V2NetBuilder::buildNetworkWithOneInput("ConvNet", in, prc_name.str())
                .addLayer("Convolution", prc_name.str(), &params, {{in}, {out}}, OC*IC*KH*KW*sizeof(data_t), OC*sizeof(data_t))
                .finish(false);
    }
};

TEST_P(IO_BlobTest, CheckDefaultValues_In) {
    auto infos = net.getInputsInfo();
    ASSERT_EQ(1, infos.size());

    auto in_info = infos["Input0"];
    ASSERT_EQ(in_info->getLayout(), NCHW);
    ASSERT_EQ(in_info->getPrecision(), Precision::FP32);
    ASSERT_EQ(in_info->getTensorDesc().getDims(), SizeVector({2,3,25,25}));

    auto ex_net = ie->LoadNetwork(net, deviceName, deviceConfig);
    auto inf_req = ex_net.CreateInferRequestPtr();
    auto blob = inf_req->GetBlob("Input0");

    ASSERT_EQ(blob->getTensorDesc().getLayout(), Layout::NCHW);
    ASSERT_EQ(blob->getTensorDesc().getPrecision(), Precision::FP32);
    ASSERT_EQ(blob->getTensorDesc().getDims(), SizeVector({2,3,25,25}));

    auto ext_blob = make_shared_blob<float>({Precision::FP32, {2, 3, 25, 25}, Layout::NCHW});
    ext_blob->allocate();
    ASSERT_NO_THROW(inf_req->SetBlob("Input0", ext_blob));
}

TEST_P(IO_BlobTest, CheckDefaultValues_Out) {
    auto infos = net.getOutputsInfo();
    ASSERT_EQ(1, infos.size());

    auto out_info = infos["Convolution1"];
    ASSERT_EQ(out_info->getLayout(), NCHW);
    ASSERT_EQ(out_info->getPrecision(), Precision::FP32);
    ASSERT_EQ(out_info->getTensorDesc().getDims(), SizeVector({2,3,25,25}));

    auto ex_net = ie->LoadNetwork(net, deviceName, deviceConfig);
    auto inf_req = ex_net.CreateInferRequestPtr();
    auto blob = inf_req->GetBlob("Convolution1");

    ASSERT_EQ(blob->getTensorDesc().getLayout(), Layout::NCHW);
    ASSERT_EQ(blob->getTensorDesc().getPrecision(), Precision::FP32);
    ASSERT_EQ(blob->getTensorDesc().getDims(), SizeVector({2,3,25,25}));

    auto ext_blob = make_shared_blob<float>({Precision::FP32, {2,3,25,25}, Layout::NCHW});
    ext_blob->allocate();
    ASSERT_NO_THROW(inf_req->SetBlob("Convolution1", ext_blob));
}

TEST_P(IO_BlobTest, DISABLED_NoAcceptBadBlobs_In) {
    auto ex_net = ie->LoadNetwork(net, deviceName, deviceConfig);
    auto inf_req = ex_net.CreateInferRequestPtr();

    auto in_blob_0 = make_shared_blob<float>({Precision::FP32, {2, 3, 25, 25},     Layout::NCHW}); // not allocated
    auto in_blob_1 = make_shared_blob<float>({Precision::FP32, {2, 3, 25, 25},     Layout::NHWC}); // wrong layout
    auto in_blob_2 = make_shared_blob<float>({Precision::FP32, {1, 1, 3*25*25, 2}, Layout::NCHW}); // wrong dims
    auto in_blob_3 = make_shared_blob<float>({Precision::FP32, {2*3*25*25},        Layout::C});    // wrong dims num
    auto in_blob_4 = make_shared_blob<uint8_t>({Precision::U8, {2, 3, 25, 25},     Layout::NCHW}); // wrong precision

    // in_blob_0 - is not allocated
    in_blob_1->allocate();
    in_blob_2->allocate();
    in_blob_3->allocate();
    in_blob_4->allocate();

    ASSERT_THROW(inf_req->SetBlob("Input0", in_blob_0), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Input0", in_blob_1), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Input0", in_blob_2), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Input0", in_blob_3), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Input0", in_blob_4), std::exception);
}

TEST_P(IO_BlobTest, DISABLED_NoAcceptBadBlobs_Out) {
    auto ex_net = ie->LoadNetwork(net, deviceName, deviceConfig);
    auto inf_req = ex_net.CreateInferRequestPtr();

    auto in_blob_0 = make_shared_blob<float>({Precision::FP32, {2, 3, 25, 25},     Layout::NCHW}); // not allocated
    auto in_blob_1 = make_shared_blob<float>({Precision::FP32, {2, 3, 25, 25},     Layout::NHWC}); // wrong layout
    auto in_blob_2 = make_shared_blob<float>({Precision::FP32, {1, 1, 3*25*25, 2}, Layout::NCHW}); // wrong dims
    auto in_blob_3 = make_shared_blob<float>({Precision::FP32, {2*3*25*25},        Layout::C});    // wrong dims num
    auto in_blob_4 = make_shared_blob<uint8_t>({Precision::U8, {2, 3, 25, 25},     Layout::NCHW}); // wrong precision

    // in_blob_0 - is not allocated
    in_blob_1->allocate();
    in_blob_2->allocate();
    in_blob_3->allocate();
    in_blob_4->allocate();

    ASSERT_THROW(inf_req->SetBlob("Convolution1", in_blob_0), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Convolution1", in_blob_1), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Convolution1", in_blob_2), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Convolution1", in_blob_3), std::exception);
    ASSERT_THROW(inf_req->SetBlob("Convolution1", in_blob_4), std::exception);
}

static auto params = ::testing::Values(Precision::FP32);  // network precision

static auto params_myriad = ::testing::Values(Precision::FP16);  // network precision

#define PLUGING_CASE(_device, _test, _params) \
    INSTANTIATE_TEST_CASE_P(_device##_run, _test, ::testing::Combine(::testing::Values(#_device), _params) )

#define PLUGING_CASE_WITH_SUFFIX(_device, _suffix, _test, _params) \
    INSTANTIATE_TEST_CASE_P(_device##_run##_suffix, _test, ::testing::Combine(::testing::Values(#_device), _params) )
