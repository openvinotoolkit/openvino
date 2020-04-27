// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include "tests_vpu_common.hpp"

#include <cpp/ie_executable_network.hpp>

#include <limits>

using namespace vpu;
using namespace InferenceEngine;

using VPU_AddVpuScaleTest = GraphTransformerTest;

// TEST_F(VPU_AddVpuScaleTest, CanAddVpuScaleToNetwork) {
//     InitCompileEnv();

//     auto& env = CompileEnv::get();
//     CompilationConfig config{};
//     config.irWithVpuScalesDir = "/";
//     env.updateConfig(config);

//     Builder::Network builder("network");
//     Builder::FullyConnectedLayer fcBuilder("FullyConnected");

//     fcBuilder.setOutputNum(1024 * 1);
//     SizeVector inputDims = {1, 2, 16, 16};

//     idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(inputDims)));

//     Blob::Ptr blob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1024, 2, 16, 16}, Layout::OIHW));
//     blob->allocate();

//     idx_t weightsId = builder.addLayer(Builder::ConstLayer("weights").setData(blob));
//     layerId = builder.addLayer({{layerId}, {weightsId}}, fcBuilder);
//     builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("output"));

//     auto network = Builder::convertToICNNNetwork(builder.build());

//     CNNLayerPtr layer;
//     network->getLayerByName("FullyConnected", layer, nullptr);

//     EXPECT_EQ(layer->params.find("vpu_scale"), layer->params.end());

//     auto model = frontEnd->buildInitialModel(*network);

//     auto middleEnd = passManager->buildMiddleEnd();

//     middleEnd->run(model);

//     EXPECT_NE(layer->params.find("vpu_scale"), layer->params.end());
// }

// TEST_F(VPU_AddVpuScaleTest, VpuScaleFromIrChangesWeights) {
//     InitCompileEnv();
//     const auto& env = CompileEnv::get();
//     CompilationConfig config{};
//     config.irWithVpuScalesDir = "/";
//     env.updateConfig(config);

//     Builder::Network netBuilder("network");

//     Blob::Ptr weightsBlob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1, 1, 1, 1}, Layout::NCHW));
//     weightsBlob->allocate();
//     auto buf = weightsBlob->buffer().as<ie_fp16*>();

//     for (size_t i = 0; i < weightsBlob->size(); ++i) {
//         buf[i] = PrecisionUtils::f32tof16(1.f);
//     }

//     idx_t layerId = netBuilder.addLayer(Builder::InputLayer("input").setPort(Port({1, 1, 1, 1})));
//     size_t weightsId = netBuilder.addLayer(Builder::ConstLayer("weights").setData(weightsBlob));

//     const auto convBuilder = Builder::ConvolutionLayer("Convolution").setStrides({1, 1}).setKernel({1, 1})
//             .setOutDepth(1).setInputPort(Port({1, 1, 1, 1}));

//     layerId = netBuilder.addLayer({{layerId}, {weightsId}}, convBuilder);
//     netBuilder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("output"));

//     auto network = Builder::convertToICNNNetwork(netBuilder.build());

//     CNNLayerPtr layer;
//     network->getLayerByName("Convolution", layer, nullptr);

//     auto model = frontEnd->buildInitialModel(*network);
//     auto middleEnd = passManager->buildMiddleEnd();

//     auto checkWeightWasChanged = [this, network, layer](const float scale) {
//         layer->params["vpu_scale"] = toString(scale);
//         auto model = frontEnd->buildInitialModel(*network);
//         auto middleEnd = passManager->buildMiddleEnd();
//         middleEnd->run(model);
//         for (const auto& stage : model->getStages()) {
//             if (stage->name() == "Convolution") {
//                 auto content = stage->input(1)->content()->get<ie_fp16>();
//                 EXPECT_EQ(scale, PrecisionUtils::f16tof32(content[0]));
//             }
//         }
//     };

//     checkWeightWasChanged(32);
//     checkWeightWasChanged(64);

//     const auto maxVal = std::numeric_limits<float>::infinity();
//     layer->params["vpu_scale"] = toString(maxVal);
//     model = frontEnd->buildInitialModel(*network);
//     middleEnd = passManager->buildMiddleEnd();
//     middleEnd->run(model);

//     for (const auto& stage : model->getStages()) {
//         if (stage->name() == "Convolution") {
//             EXPECT_EQ(stage->attrs().get<float>("scaleFactor"), maxVal);
//         }
//     }
// }
