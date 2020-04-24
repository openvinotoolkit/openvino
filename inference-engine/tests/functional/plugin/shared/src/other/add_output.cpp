// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "other/add_output.hpp"

// TODO: Replace IRBuilder with NGraph when it supports Memory Layer
std::string AddOutputTestsCommonClass::generate_model() {
    CommonTestUtils::IRBuilder_v6 test_model_builder("model");

    auto precision = InferenceEngine::Precision::FP32;

    auto Memory_1_layer =
        test_model_builder.AddLayer("Memory_1", "Memory", precision, {{"id", "r_1-3"}, {"index", "1"}, {"size", "2"}})
            .AddOutPort({1, 200})
            .getLayer();
    auto Input_2_layer = test_model_builder.AddLayer("Input_2", "input", precision).AddOutPort({1, 200}).getLayer();
    auto Eltwise_3_layer = test_model_builder.AddLayer("Eltwise_3", "Eltwise", precision, {{"operation", "mul"}})
                               .AddInPort({1, 200})
                               .AddInPort({1, 200})
                               .AddOutPort({1, 200})
                               .getLayer();

    auto Activation_4_layer =
        test_model_builder.AddLayer("Activation_4", "Activation", precision, {{"type", "sigmoid"}})
            .AddInPort({1, 200})
            .AddOutPort({1, 200})
            .getLayer();
    auto Memory_5_layer =
        test_model_builder.AddLayer("Memory_5", "Memory", precision, {{"id", "r_1-3"}, {"index", "0"}, {"size", "2"}})
            .AddInPort({1, 200})
            .getLayer();

    test_model_builder.AddEdge(Memory_1_layer.out(0), Eltwise_3_layer.in(0));
    test_model_builder.AddEdge(Input_2_layer.out(0), Eltwise_3_layer.in(1));
    test_model_builder.AddEdge(Eltwise_3_layer.out(0), Activation_4_layer.in(0));
    test_model_builder.AddEdge(Activation_4_layer.out(0), Memory_5_layer.in(0));

    auto serial = test_model_builder.serialize();

    return serial;
}

std::string AddOutputTestsCommonClass::getTestCaseName(
    testing::TestParamInfo<std::tuple<std::string, std::string>> obj) {
    std::string layer;
    std::string engine;

    std::tie(layer, engine) = obj.param;
    return layer + "_" + engine;
}

void AddOutputTestsCommonClass::run_test() {
    std::string layer_name;
    std::string engine_type;

    std::tie(layer_name, engine_type) = this->GetParam();

    auto model = this->generate_model();

    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNet;

    auto null_blob = CommonTestUtils::getWeightsBlob(0);
    network = ie.ReadNetwork(model, null_blob);
    network.addOutput(layer_name);
    executableNet = ie.LoadNetwork(network, engine_type);

    auto outputs = executableNet.GetOutputsInfo();

    auto layer_output = outputs[layer_name];

    ASSERT_EQ(true, layer_output && "layer not found in outputs");
}
