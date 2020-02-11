// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include "graph_transformer_tests.hpp"
#include "tests_vpu_common.hpp"

using namespace vpu;
using namespace InferenceEngine;

using VPU_AddVpuScaleTest = GraphTransformerTest;

TEST_F(VPU_AddVpuScaleTest, CanAddVpuScaleToNetwork) {
    InitCompileEnv();

    auto& env = CompileEnv::get();
    CompilationConfig config{};
    config.irWithVpuScalesDir = "/";
    env.updateConfig(config);

    Builder::Network builder("network");
    Builder::FullyConnectedLayer fcBuilder("FullyConnected");

    fcBuilder.setOutputNum(1024 * 1);
    SizeVector inputDims = {1, 2, 16, 16};

    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(inputDims)));

    Blob::Ptr blob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1024, 2, 16, 16}, Layout::OIHW));
    blob->allocate();

    idx_t weightsId = builder.addLayer(Builder::ConstLayer("weights").setData(blob));
    layerId = builder.addLayer({{layerId}, {weightsId}}, fcBuilder);
    builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("output"));

    auto network = Builder::convertToICNNNetwork(builder.build());

    CNNLayerPtr layer;
    network->getLayerByName("FullyConnected", layer, nullptr);

    EXPECT_EQ(layer->params.find("vpu_scale"), layer->params.end());

    auto model = frontEnd->buildInitialModel(*network);

    auto middleEnd = passManager->buildMiddleEnd();

    middleEnd->run(model);

    EXPECT_NE(layer->params.find("vpu_scale"), layer->params.end());
}

TEST_F(VPU_AddVpuScaleTest, ScaleFactorDoesNotChange) {
    InitCompileEnv();

    const auto& env = CompileEnv::get();
    CompilationConfig config{};
    config.irWithVpuScalesDir = "/";
    env.updateConfig(config);

    Builder::Network builder("network");
    Builder::FullyConnectedLayer fcBuilder("FullyConnected");

    fcBuilder.setOutputNum(1024 * 1);
    SizeVector inputDims = {1, 2, 16, 16};

    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(inputDims)));

    Blob::Ptr blob = make_shared_blob<ie_fp16>(TensorDesc(Precision::FP16, {1024, 2, 16, 16}, Layout::OIHW));
    blob->allocate();

    idx_t weightsId = builder.addLayer(Builder::ConstLayer("weights").setData(blob));
    layerId = builder.addLayer({{layerId}, {weightsId}}, fcBuilder);
    builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("output"));

    auto network = Builder::convertToICNNNetwork(builder.build());

    CNNLayerPtr layer;
    network->getLayerByName("FullyConnected", layer, nullptr);

    auto max_val = std::numeric_limits<float>::infinity();
    layer->params["vpu_scale"] = toString(max_val);

    auto model = frontEnd->buildInitialModel(*network);

    auto middleEnd = passManager->buildMiddleEnd();

    middleEnd->run(model);

    for (const auto& stage : model->getStages()) {
        if (stage->type() == StageType::MyriadXHwOp) {
            EXPECT_EQ(stage->attrs().get<float>("scaleFactor"), max_val);
        }
    }
}
