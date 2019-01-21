// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include <gtest/gtest.h>
//#include "mkldnn_plugin/mkldnn_layers.h"
//
//using namespace std;
//
//class MKLDNNLayersTests : public ::testing::Test {
//protected:
//    virtual void TearDown() override{
//    }
//
//    virtual void SetUp() override{
//    }
//
//};
//
//TEST_F(MKLDNNLayersTests, canCreateContext) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> dl ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateConvLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    InferenceEngine::TBlob<float>::Ptr blobPtr(new InferenceEngine::TBlob<float>());
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context(blobPtr, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::ConvolutionLayer convLayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {0, 0, 0, 0};
//
//    convLayer.insData.push_back(dPtr);
//    convLayer.outData.push_back(dPtr);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&convLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::ConvLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateLRNLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::NormLayer normLayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {1, 1, 27, 27};
//
//    normLayer.insData.push_back(dPtr);
//    normLayer.outData.push_back(dPtr);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&normLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::LRNLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreatePoolingLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::PoolingLayer poolingLayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {1, 1, 27, 27};
//
//    poolingLayer.insData.push_back(dPtr);
//    poolingLayer.outData.push_back(dPtr);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&poolingLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::PoolingLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateSplitLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::SplitLayer splitLayer({});
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&splitLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::SplitLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateConcatLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::ConcatLayer concatLayer({});
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&concatLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::ConcatLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateFullyConnectedLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    InferenceEngine::TBlob<float>::Ptr blobPtr(new InferenceEngine::TBlob<float>());
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context(blobPtr, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::FullyConnectedLayer fcLayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {0, 0, 0, 0};
//    InferenceEngine::DataPtr dPtr2(new InferenceEngine::Data("testData2"));
//    dPtr2->dims = {0, 0};
//
//    fcLayer.insData.push_back(dPtr);
//    fcLayer.outData.push_back(dPtr2);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&fcLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::FullyConnectedLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateSoftMaxLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::SoftMaxLayer softmaxLayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {0, 0, 0, 0};
//    InferenceEngine::DataPtr dPtr2(new InferenceEngine::Data("testData2"));
//    dPtr2->dims = {0, 0};
//
//    softmaxLayer.insData.push_back(dPtr);
//    softmaxLayer.outData.push_back(dPtr2);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&softmaxLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::SoftMaxLayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canCreateReLULayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::ReLULayer reLULayer({});
//    InferenceEngine::DataPtr dPtr(new InferenceEngine::Data("testData"));
//    dPtr->dims = {1, 1, 27, 27};
//
//    reLULayer.insData.push_back(dPtr);
//    reLULayer.outData.push_back(dPtr);
//    unique_ptr<MKLDNNPlugin::Layer> dl ( MKLDNNPlugin::LayerRegistry::CreateLayer(&reLULayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())));
//
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::ReLULayer*>(dl.get()));
//}
//
//TEST_F(MKLDNNLayersTests, canNotCreateCNNLayer) {
//    std::vector<float> sd;
//    std::vector<float> dd;
//    std::vector<size_t> ds;
//    unique_ptr<MKLDNNPlugin::Context> ctx ( new MKLDNNPlugin::Context({}, mkldnn::engine(mkldnn::engine::cpu, 0), &sd, &dd, &ds));
//    ASSERT_NE(nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get()));
//
//    InferenceEngine::CNNLayer cnnLayer({});
//    EXPECT_THROW(MKLDNNPlugin::LayerRegistry::CreateLayer(&cnnLayer, nullptr, dynamic_cast<MKLDNNPlugin::Context*>(ctx.get())) , InferenceEngine::details::InferenceEngineException);
//}
//
//TEST_F(MKLDNNLayersTests, canNotCreateLayerWithoutContext) {
//    InferenceEngine::ConvolutionLayer convLayer({});
//    EXPECT_THROW(MKLDNNPlugin::LayerRegistry::CreateLayer(&convLayer, nullptr, nullptr), InferenceEngine::details::InferenceEngineException);
//}