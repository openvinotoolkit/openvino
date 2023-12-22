// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/file_utils.hpp>

#include "cpp/ie_cnn_network.h"
#include "inference_engine.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"

using namespace InferenceEngine;

using CNNNetworkTests = ::testing::Test;

IE_SUPPRESS_DEPRECATED_START

TEST_F(CNNNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<ICNNNetwork> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedCastToICNNNetwork) {
    CNNNetwork network;
    ASSERT_THROW((void)static_cast<ICNNNetwork&>(network), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedCastToICNNNetwork) {
    const CNNNetwork network;
    ASSERT_THROW((void)static_cast<const ICNNNetwork&>(network), InferenceEngine::Exception);
}

IE_SUPPRESS_DEPRECATED_END

TEST_F(CNNNetworkTests, throwsOnInitWithNullNgraph) {
    std::shared_ptr<ngraph::Function> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getOutputsInfo(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetInputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputsInfo(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedLayerCount) {
    CNNNetwork network;
    ASSERT_THROW(network.layerCount(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetName) {
    CNNNetwork network;
    ASSERT_THROW(network.getName(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetFunction) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedGetFunction) {
    const CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedBegin) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::Exception);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedGetInputShapes) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputShapes(), InferenceEngine::Exception);
}

static std::shared_ptr<ov::Model> CNNNetworkTests_create_model() {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    param1->set_friendly_name("p1_friendly");
    param1->output(0).set_names({"p1_1", "p1_2"});
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, 224, 224});
    param2->set_friendly_name("p2_friendly");
    param2->output(0).set_names({"p2_1", "p2_2"});
    auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 224, 224});
    param3->set_friendly_name("p3_friendly");
    param3->output(0).set_names({"p3_1", "p3_2"});
    return std::make_shared<ov::Model>(ov::OutputVector{param1, param2, param3},
                                       ov::ParameterVector{param1, param2, param3});
}

TEST_F(CNNNetworkTests, throwsHasDynamicInputs) {
    auto model = CNNNetworkTests_create_model();
    CNNNetwork network(model);
    InferenceEngine::Core core;
    try {
        core.LoadNetwork(network);
        FAIL() << "LoadNetwork with dynamic inputs shall throw";
    } catch (const InferenceEngine::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("InferenceEngine::Core::LoadNetwork") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_1") == std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_2") == std::string::npos) << e.what();
    }
}

TEST_F(CNNNetworkTests, throwsHasDynamicInputs_remoteContext) {
    auto model = CNNNetworkTests_create_model();
    CNNNetwork network(model);
    InferenceEngine::Core core;
    try {
        core.LoadNetwork(network, InferenceEngine::RemoteContext::Ptr());
        FAIL() << "LoadNetwork with dynamic inputs shall throw";
    } catch (const InferenceEngine::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("InferenceEngine::Core::LoadNetwork") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_1") == std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_2") == std::string::npos) << e.what();
    }
}

TEST_F(CNNNetworkTests, throwsHasDynamicInputs_queryNetwork) {
    auto model = CNNNetworkTests_create_model();
    CNNNetwork network(model);
    InferenceEngine::Core core;
    try {
        core.QueryNetwork(network, "mock");
        FAIL() << "QueryNetwork with dynamic inputs shall throw";
    } catch (const InferenceEngine::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("InferenceEngine::Core::QueryNetwork") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p1_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_1") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p2_2") != std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_1") == std::string::npos) << e.what();
        EXPECT_TRUE(std::string(e.what()).find("p3_2") == std::string::npos) << e.what();
    }
}
