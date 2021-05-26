// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>

#include "functional_test_utils/test_model/test_model.hpp"
#include "ir_net.hpp"
// #include "common_layers_params.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include <ngraph_functions/subgraph_builders.hpp>
#include <ngraph/pass/manager.hpp>
#include "transformations/serialize.hpp"
#include "ie_ngraph_utils.hpp"

IE_SUPPRESS_DEPRECATED_START

namespace FuncTestUtils {
namespace TestModel {

/**
 * @brief generates IR files (XML and BIN files) with the test model.
 *        Passed reference vector is filled with CNN layers to validate after the network reading.
 * @param modelPath used to serialize the generated network
 * @param weightsPath used to serialize the generated weights
 * @param netPrc precision of the generated network
 * @param inputDims dims on the input layer of the generated network
 */
void generateTestModel(const std::string &modelPath,
                       const std::string &weightsPath,
                       const InferenceEngine::Precision &netPrc,
                       const InferenceEngine::SizeVector &inputDims) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>(
            modelPath, weightsPath,
            ngraph::pass::Serialize::Version::IR_V10);
    manager.run_passes(ngraph::builder::subgraph::makeMultiSingleConv(
            inputDims, InferenceEngine::details::convertPrecision(netPrc)));
}

TestModel getModelWithMemory(InferenceEngine::Precision netPrc) {
    CommonTestUtils::IRBuilder_v6 test_model_builder("model");

    auto Memory_1_layer =
            test_model_builder.AddLayer("Memory_1", "Memory", netPrc, {{"id",    "r_1-3"},
                                                                       {"index", "1"},
                                                                       {"size",  "2"}})
                    .AddOutPort({1, 200})
                    .getLayer();
    auto Input_2_layer = test_model_builder.AddLayer("Input_2", "input", netPrc).AddOutPort({1, 200}).getLayer();
    auto Eltwise_3_layer = test_model_builder.AddLayer("Eltwise_3", "Eltwise", netPrc, {{"operation", "mul"}})
            .AddInPort({1, 200})
            .AddInPort({1, 200})
            .AddOutPort({1, 200})
            .getLayer();

    auto Activation_4_layer =
            test_model_builder.AddLayer("Activation_4", "Activation", netPrc, {{"type", "sigmoid"}})
                    .AddInPort({1, 200})
                    .AddOutPort({1, 200})
                    .getLayer();
    auto Memory_5_layer =
            test_model_builder.AddLayer("Memory_5", "Memory", netPrc, {{"id",    "r_1-3"},
                                                                       {"index", "0"},
                                                                       {"size",  "2"}})
                    .AddInPort({1, 200})
                    .getLayer();

    test_model_builder.AddEdge(Memory_1_layer.out(0), Eltwise_3_layer.in(0));
    test_model_builder.AddEdge(Input_2_layer.out(0), Eltwise_3_layer.in(1));
    test_model_builder.AddEdge(Eltwise_3_layer.out(0), Activation_4_layer.in(0));
    test_model_builder.AddEdge(Activation_4_layer.out(0), Memory_5_layer.in(0));

    auto serial = test_model_builder.serialize();

    return TestModel(serial, {});
}
TestModel getModelWithMultipleMemoryConnections(InferenceEngine::Precision netPrc) {
    CommonTestUtils::IRBuilder_v6 test_model_builder("model");

    auto Memory_1_layer =
        test_model_builder.AddLayer("Memory_1", "Memory", netPrc, { {"id",    "r_1-3"},
                                                                    {"index", "1"},
                                                                    {"size",  "2"} })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Input_1_layer = test_model_builder.AddLayer("Input_1", "input", netPrc).AddOutPort({ 1, 200 }).getLayer();
    auto Eltwise_1_layer = test_model_builder.AddLayer("Eltwise_1", "Eltwise", netPrc, { {"operation", "mul"} })
        .AddInPort({ 1, 200 })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();

    auto Memory_2_layer =
        test_model_builder.AddLayer("Memory_2", "Memory", netPrc, { {"id",    "c_1-3"},
                                                                    {"index", "1"},
                                                                    {"size",  "2"} })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Eltwise_2_layer = test_model_builder.AddLayer("Eltwise_2", "Eltwise", netPrc, { {"operation", "mul"} })
        .AddInPort({ 1, 200 })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Memory_3_layer =
        test_model_builder.AddLayer("Memory_3", "Memory", netPrc, { {"id",    "c_1-3"},
                                                                   {"index", "0"},
                                                                   {"size",  "2"} })
        .AddInPort({ 1, 200 })
        .getLayer();

    auto Activation_1_layer =
        test_model_builder.AddLayer("Activation_1", "Activation", netPrc, { {"type", "sigmoid"} })
        .AddInPort({ 1, 200 })
        .AddOutPort({ 1, 200 })
        .getLayer();
    auto Memory_4_layer =
        test_model_builder.AddLayer("Memory_4", "Memory", netPrc, { {"id",    "r_1-3"},
                                                                   {"index", "0"},
                                                                   {"size",  "2"} })
        .AddInPort({ 1, 200 })
        .getLayer();

    test_model_builder.AddEdge(Memory_1_layer.out(0), Eltwise_1_layer.in(0));
    test_model_builder.AddEdge(Input_1_layer.out(0), Eltwise_1_layer.in(1));
    test_model_builder.AddEdge(Eltwise_1_layer.out(0), Eltwise_2_layer.in(1));
    test_model_builder.AddEdge(Memory_2_layer.out(0), Eltwise_2_layer.in(0));
    test_model_builder.AddEdge(Eltwise_2_layer.out(0), Memory_3_layer.in(0));
    test_model_builder.AddEdge(Eltwise_2_layer.out(0), Activation_1_layer.in(0));
    test_model_builder.AddEdge(Activation_1_layer.out(0), Memory_4_layer.in(0));

    auto serial = test_model_builder.serialize();
    return TestModel(serial, {});
}
}  // namespace TestModel
}  // namespace FuncTestUtils
