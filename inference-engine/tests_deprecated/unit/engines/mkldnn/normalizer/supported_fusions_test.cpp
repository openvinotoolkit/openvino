// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <mkldnn_plugin.h>
#include <ie_blob.h>
#include <ie_precision.hpp>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "common_test_utils/xml_net_builder/xml_net_builder.hpp"
#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/data_utils.hpp"


struct conv_eltwise_params {
    std::vector<size_t> in1;
    std::vector<size_t> in2;

    CommonTestUtils::conv_common_params conv;
    CommonTestUtils::eltwise_common_params eltwise;
};

struct in_conv_in_conv_eltwise_params {
    std::vector<size_t> in1;
    std::vector<size_t> in2;

    CommonTestUtils::conv_common_params conv1;
    CommonTestUtils::conv_common_params conv2;
    CommonTestUtils::eltwise_common_params eltwise;
};

struct conv_conv_eltwise_conv_pooling_params {
    std::vector<size_t> in1;
    std::vector<size_t> in2;

    CommonTestUtils::conv_common_params conv1;
    CommonTestUtils::conv_common_params conv2;
    CommonTestUtils::conv_common_params conv3;
    CommonTestUtils::eltwise_common_params eltwise;
    CommonTestUtils::pool_common_params pool;
};

class ConvSum: public TestsCommon, public ::testing::WithParamInterface<conv_eltwise_params> {
    std::string getModel(conv_eltwise_params p) {
        std::string precision = "FP32";
        std::vector<size_t> convOutShape(p.in1.size());
        getConvOutShape(p.in1, p.conv, convOutShape);

        std::vector<float> min_stat(p.in1[1]);
        std::vector<float> max_stat(p.in1[1]);
        CommonTestUtils::fill_data_sine(min_stat.data(), p.in1[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(max_stat.data(), p.in1[1], 1, 1, -1);
        CommonTestUtils::Statistic in_stat = {min_stat, max_stat};
        std::vector<float> conv_min_stat(convOutShape[1]);
        std::vector<float> conv_max_stat(convOutShape[1]);
        CommonTestUtils::fill_data_sine(conv_min_stat.data(), convOutShape[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(conv_max_stat.data(), convOutShape[1], 1, 1, -1);
        CommonTestUtils::Statistic conv_stat = {conv_min_stat, conv_max_stat};

        std::map<std::string, std::string> elt_params = {
                {"operation", "sum"}
        };
        std::vector<std::pair<std::string, std::string>> edges = { {"0,0", "2,2"}, {"2,3", "3,4"}, {"1,1", "3,5"} };

        return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
                "Fusion_conv_sum", p.in1, precision, in_stat)
                .addInputLayer(precision, convOutShape, in_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape}}, p.conv, conv_stat)
                .addLayer("Eltwise", precision, &elt_params, {{convOutShape, convOutShape}, {convOutShape}}, 0, 0, "data", "", conv_stat)
                .finish(&edges);
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_eltwise_params p = ::testing::WithParamInterface<conv_eltwise_params>::GetParam();
            std::string model = getModel(p);
            printf("model:\n%s", model.c_str());

            InferenceEngine::Core ie;
            auto network = ie.ReadNetwork(model, getConvWeightsBlob(p.in1, p.conv));
            std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
            InferenceEngine::IExecutableNetwork::Ptr exeNetwork1;
            ASSERT_NO_THROW(score_engine->LoadNetwork(exeNetwork1, network, {}));

            auto conv = network.getLayerByName("Convolution2");
            auto eltwise = network.getLayerByName("Eltwise3");

            ASSERT_EQ(conv->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class ConvSumReLU: public TestsCommon, public ::testing::WithParamInterface<conv_eltwise_params> {
    std::string getModel(conv_eltwise_params p) {
        std::string precision = "FP32";
        std::vector<size_t> convOutShape(p.in1.size());
        getConvOutShape(p.in1, p.conv, convOutShape);

        std::vector<float> min_stat(p.in1[1]);
        std::vector<float> max_stat(p.in1[1]);
        CommonTestUtils::fill_data_sine(min_stat.data(), p.in1[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(max_stat.data(), p.in1[1], 1, 1, -1);
        CommonTestUtils::Statistic in_stat = {min_stat, max_stat};
        std::vector<float> conv_min_stat(convOutShape[1]);
        std::vector<float> conv_max_stat(convOutShape[1]);
        CommonTestUtils::fill_data_sine(conv_min_stat.data(), convOutShape[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(conv_max_stat.data(), convOutShape[1], 1, 1, -1);
        CommonTestUtils::Statistic conv_stat = {conv_min_stat, conv_max_stat};

        std::map<std::string, std::string> elt_params = {
                {"operation", "sum"}
        };
        std::map<std::string, std::string> relu_params = {};
        std::vector<std::pair<std::string, std::string>> edges = { {"0,0", "2,2"}, {"2,3", "3,4"}, {"1,1", "3,5"}, {"3,6", "4,7"} };
        return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
                "Fusion_conv_sum", p.in1, precision, in_stat)
                .addInputLayer(precision, convOutShape, in_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape}}, p.conv, conv_stat)
                .addLayer("Eltwise", precision, &elt_params, {{convOutShape, convOutShape}, {convOutShape}}, 0, 0, "data", "", conv_stat)
                .addLayer("ReLU", precision, &relu_params, {{convOutShape, convOutShape}, {convOutShape}}, 0, 0, "data", "", conv_stat)
                .finish(&edges);
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_eltwise_params p = ::testing::WithParamInterface<conv_eltwise_params>::GetParam();
            std::string model = getModel(p);
            printf("model:\n%s", model.c_str());

            Core ie;
            auto network = ie.ReadNetwork(model, getConvWeightsBlob(p.in1, p.conv));

            std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
            InferenceEngine::IExecutableNetwork::Ptr exeNetwork1;
            ASSERT_NO_THROW(score_engine->LoadNetwork(exeNetwork1, network, { }));

            auto conv = network.getLayerByName("Convolution2");
            auto eltwise = network.getLayerByName("Eltwise3");
            auto relu4 = network.getLayerByName("ReLU4");

            ASSERT_EQ(conv->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(relu4->precision, InferenceEngine::Precision::I8);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class ConvConvSum: public TestsCommon, public ::testing::WithParamInterface<conv_eltwise_params> {
    std::string getModel(conv_eltwise_params p) {
        std::string precision = "FP32";
        std::vector<size_t> convOutShape(p.in1.size());
        getConvOutShape(p.in1, p.conv, convOutShape);

        std::vector<float> min_stat(p.in1[1]);
        std::vector<float> max_stat(p.in1[1]);
        CommonTestUtils::fill_data_sine(min_stat.data(), p.in1[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(max_stat.data(), p.in1[1], 1, 1, -1);
        CommonTestUtils::Statistic in_stat = {min_stat, max_stat};
        std::vector<float> conv_min_stat(convOutShape[1]);
        std::vector<float> conv_max_stat(convOutShape[1]);
        CommonTestUtils::fill_data_sine(conv_min_stat.data(), convOutShape[1], -1, 1, 1);
        CommonTestUtils::fill_data_sine(conv_max_stat.data(), convOutShape[1], 1, 1, -1);
        CommonTestUtils::Statistic conv_stat = {conv_min_stat, conv_max_stat};

        std::map<std::string, std::string> elt_params = {
                {"operation", "sum"}
        };
        std::vector<std::pair<std::string, std::string>> edges = { {"0,0", "2,2"}, {"2,3", "4,6"}, {"1,1", "3,4"}, {"3,5", "4,7"} };
        return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
                "Fusion_conv_sum", p.in1, precision, in_stat)
                .addInputLayer(precision, p.in1, in_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape}}, p.conv, conv_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape}}, p.conv, conv_stat)
                .addLayer("Eltwise", precision, &elt_params, {{convOutShape, convOutShape}, {convOutShape}}, 0, 0, "data", "", conv_stat)
                .finish(&edges);
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_eltwise_params p = ::testing::WithParamInterface<conv_eltwise_params>::GetParam();
            std::string model = getModel(p);
            printf("model:\n%s", model.c_str());

            Core ie;
            auto network = ie.ReadNetwork(model, getConvWeightsBlob(p.in1, p.conv));

            std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
            InferenceEngine::IExecutableNetwork::Ptr exeNetwork1;
            ASSERT_NO_THROW(score_engine->LoadNetwork(exeNetwork1, network, { }));

            auto conv2 = network.getLayerByName("Convolution2");
            auto conv3 = network.getLayerByName("Convolution3");
            auto eltwise = network.getLayerByName("Eltwise3");

            ASSERT_EQ(conv2->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv2->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class ConvConvSumReLU: public TestsCommon, public ::testing::WithParamInterface<in_conv_in_conv_eltwise_params> {
    std::string getModel(in_conv_in_conv_eltwise_params p) {
        std::string precision = "FP32";
        std::vector<size_t> convOutShape1(p.in1.size());
        std::vector<size_t> convOutShape2(p.in2.size());
        getConvOutShape(p.in1, p.conv1, convOutShape1);
        getConvOutShape(p.in2, p.conv2, convOutShape2);

        CommonTestUtils::Statistic in1_stat, in2_stat, conv1_stat, conv2_stat;
        fillStatistic(in1_stat, p.in1[1], -2, 2);
        fillStatistic(in2_stat, p.in2[1], -2, 2);
        fillStatistic(conv1_stat, p.conv1.out_c, -2, 2);
        fillStatistic(conv2_stat, p.conv2.out_c, -2, 2);

        std::map<std::string, std::string> elt_params = {
                {"operation", "sum"}
        };
        std::map<std::string, std::string> relu_params = {};
        std::vector<std::pair<std::string, std::string>> edges = { {"0,0", "2,2"}, {"2,3", "4,6"}, {"1,1", "3,4"}, {"3,5", "4,7"}, {"4,8", "5,9"} };
        return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
                "Fusion_conv_sum", p.in1, precision, in1_stat)
                .addInputLayer(precision, p.in2, in2_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape1}}, p.conv1, conv1_stat)
                .convolutionLayer(precision, {{p.in2}, {convOutShape2}}, p.conv2, conv2_stat)
                .addLayer("Eltwise", precision, &elt_params, {{convOutShape1, convOutShape2}, {convOutShape1}}, 0, 0, "data", "", conv1_stat)
                .addLayer("ReLU", precision, &relu_params, {{convOutShape1}, {convOutShape1}}, 0, 0, "data", "", conv1_stat)
                .finish(&edges);
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            in_conv_in_conv_eltwise_params p = ::testing::WithParamInterface<in_conv_in_conv_eltwise_params>::GetParam();
            std::string model = getModel(p);
            printf("model:\n%s", model.c_str());

            Core ie;
            size_t weight_size = getConvWeightsSize(p.in1, p.conv1, "FP32") + getConvBiasesSize(p.conv1, "FP32") +
                                 getConvWeightsSize(p.in2, p.conv2, "FP32") + getConvBiasesSize(p.conv2, "FP32");
            auto network = ie.ReadNetwork(model, CommonTestUtils::getWeightsBlob(weight_size));

            std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
            InferenceEngine::IExecutableNetwork::Ptr exeNetwork1;
            ASSERT_NO_THROW(score_engine->LoadNetwork(exeNetwork1, network, { }));

            auto conv2 = network.getLayerByName("Convolution2");
            auto conv3 = network.getLayerByName("Convolution3");
            auto eltwise = network.getLayerByName("Eltwise3");
            auto relu5 = network.getLayerByName("ReLU5");

            ASSERT_EQ(conv2->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv2->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(relu5->precision, InferenceEngine::Precision::I8);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class ConvConvSumReLUPoolConv: public TestsCommon, public ::testing::WithParamInterface<conv_conv_eltwise_conv_pooling_params> {
    std::string getModel(conv_conv_eltwise_conv_pooling_params p) {
        std::string precision = "FP32";
        std::vector<size_t> convOutShape1(p.in1.size());
        std::vector<size_t> convOutShape2(p.in2.size());
        std::vector<size_t> convOutShape3(p.in1.size());
        std::vector<size_t> poolOutShape(p.in2.size());
        getConvOutShape(p.in1, p.conv1, convOutShape1);
        getConvOutShape(p.in2, p.conv2, convOutShape2);
        getConvOutShape(convOutShape1, p.conv3, convOutShape3);
        getPoolOutShape(convOutShape1, p.pool, poolOutShape);

        CommonTestUtils::Statistic in1_stat, in2_stat, conv1_stat, conv2_stat, conv3_stat, pool_stat;
        fillStatistic(in1_stat, p.in1[1], -2.f, 2.f);
        fillStatistic(in2_stat, p.in2[1], -2.f, 2.f);
        fillStatistic(conv1_stat, p.conv1.out_c, -2.f, 2.f);
        fillStatistic(conv2_stat, p.conv2.out_c, -2.f, 2.f);
        fillStatistic(conv3_stat, p.conv3.out_c, -2.f, 2.f);
        fillStatistic(pool_stat, poolOutShape[1], 0.f, 3.f);

        std::map<std::string, std::string> elt_params = {
                {"operation", "sum"}
        };
        std::map<std::string, std::string> relu_params = {};
        std::vector<std::pair<std::string, std::string>> edges = { {"0,0", "2,2"},
                                                                   {"2,3", "4,6"},
                                                                   {"1,1", "3,4"},
                                                                   {"3,5", "4,7"},
                                                                   {"4,8", "5,9"},
                                                                   {"5,10", "7,13"},
                                                                   {"4,8", "6,11"} };
        return CommonTestUtils::DefaultNetBuilder::buildNetworkWithOneInput(
                "Fusion_conv_sum", p.in1, precision, in1_stat)
                .addInputLayer(precision, p.in2, in2_stat)
                .convolutionLayer(precision, {{p.in1}, {convOutShape1}}, p.conv1, conv1_stat)
                .convolutionLayer(precision, {{p.in2}, {convOutShape2}}, p.conv2, conv2_stat)
                .addLayer("Eltwise", precision, &elt_params, {{convOutShape1, convOutShape2}, {convOutShape1}}, 0, 0, "data", "", conv1_stat)
                .addLayer("ReLU", precision, &relu_params, {{convOutShape1}, {convOutShape1}}, 0, 0, "data", "", pool_stat)
                .convolutionLayer(precision, {{convOutShape1}, {convOutShape3}}, p.conv3, conv3_stat)
                .addLayer("Pooling", precision, &relu_params, {{convOutShape1}, {poolOutShape}}, 0, 0, "data", "", pool_stat)
                .finish(&edges);
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_conv_eltwise_conv_pooling_params p =
                    ::testing::WithParamInterface<conv_conv_eltwise_conv_pooling_params>::GetParam();
            std::string model = getModel(p);
            printf("model:\n%s", model.c_str());

            Core ie;
            std::vector<size_t> convOutShape3(p.in1.size());
            size_t weight_size = getConvWeightsSize(p.in1, p.conv1, "FP32") + getConvBiasesSize(p.conv1, "FP32") +
                                 getConvWeightsSize(p.in2, p.conv2, "FP32") + getConvBiasesSize(p.conv2, "FP32") +
                                 getConvWeightsSize(convOutShape3, p.conv3, "FP32") + getConvBiasesSize(p.conv3, "FP32");
            auto network = ie.ReadNetwork(model, CommonTestUtils::getWeightsBlob(weight_size));

            std::shared_ptr<MKLDNNPlugin::Engine> score_engine(new MKLDNNPlugin::Engine());
            InferenceEngine::IExecutableNetwork::Ptr exeNetwork1;
            ASSERT_NO_THROW(score_engine->LoadNetwork(exeNetwork1, network, {}));

            auto conv2 = network.getLayerByName("Convolution2");
            auto conv3 = network.getLayerByName("Convolution3");
            auto eltwise = network.getLayerByName("Eltwise3");
            auto relu5 = network.getLayerByName("ReLU5");

            ASSERT_EQ(conv2->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv2->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(conv3->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->precision, InferenceEngine::Precision::I8);
            ASSERT_EQ(eltwise->outData[0]->getPrecision(), InferenceEngine::Precision::I8);
            ASSERT_EQ(relu5->precision, InferenceEngine::Precision::I8);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};


// there is no o-scale in Input1
TEST_P(ConvSum, DISABLED_TestsNormalizerSupportedFusions) {}
INSTANTIATE_TEST_CASE_P(
        TestsNormalizerSupportedFusions, ConvSum,
        ::testing::Values(
                conv_eltwise_params{{1, 16, 4, 4}, {1, 16, 4, 4},
                                    { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                    {"sum", {}} }
        ));

TEST_P(ConvSumReLU, DISABLED_TestsNormalizerSupportedFusions) {}
INSTANTIATE_TEST_CASE_P(
        TestsNormalizerSupportedFusions, ConvSumReLU,
        ::testing::Values(
                conv_eltwise_params{{1, 16, 4, 4},  {1, 16, 4, 4},
                                    { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                    {"sum", {}} }
        ));

// there is no oi-scale in Convolution3
TEST_P(ConvConvSum, DISABLED_TestsNormalizerSupportedFusions) {}
INSTANTIATE_TEST_CASE_P(
        TestsNormalizerSupportedFusions, ConvConvSum,
        ::testing::Values(
                conv_eltwise_params{{1, 16, 4, 4}, {1, 16, 4, 4},
                                    { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                    {"sum", {}} }
        ));

TEST_P(ConvConvSumReLU, DISABLED_TestsNormalizerSupportedFusions) {}
INSTANTIATE_TEST_CASE_P(
        TestsNormalizerSupportedFusions, ConvConvSumReLU,
        ::testing::Values(
                in_conv_in_conv_eltwise_params{{1, 16, 4, 4}, {1, 16, 4, 4},
                                               { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                               { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                               {"sum", {}} },
                in_conv_in_conv_eltwise_params{{1, 48, 40, 20}, {1, 32, 40, 20},
                                               { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, true, "I8" },
                                               { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, true, "I8" },
                                               {"sum", {}} }
        ));

TEST_P(ConvConvSumReLUPoolConv, DISABLED_TestsNormalizerSupportedFusions) {}
INSTANTIATE_TEST_CASE_P(
        TestsNormalizerSupportedFusions, ConvConvSumReLUPoolConv,
        ::testing::Values(
                conv_conv_eltwise_conv_pooling_params{{1, 16, 4, 4}, {1, 16, 4, 4},
                                                      { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                                      { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                                      { {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 32, true, "I8" },
                                                      {"sum", {}},
                                                      { {1, 1}, {1, 1}, {0, 0}, {0, 0} } }
        ));

