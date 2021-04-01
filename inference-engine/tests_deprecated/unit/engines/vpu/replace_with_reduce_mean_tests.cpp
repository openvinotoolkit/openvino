// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/stub_stage.hpp>

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_ReplaceWithReduceMeanTest : public GraphTransformerTest {
protected:
    PassSet pipeline;
    Model model;
public:
    void InitPipeline() {
        pipeline = PassSet();
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->replaceWithReduceMean());
        pipeline.addPass(passManager->dumpModel("replaceWithReduceMean"));
    }
};

TEST_F(VPU_ReplaceWithReduceMeanTest, ReplaceAvgPoolWithReduceMean) {
    InitCompileEnv();

    model = CreateModel();

    int inputW = 112;
    int inputH = 112;

    auto input = model->addInputData("Input", DataDesc(DataType::FP16, DimsOrder::NCHW, {inputW, inputH, 32, 1}));
    model->attrs().set<int>("numInputs", 1);

    auto output = model->addOutputData("Output", DataDesc(DataType::FP16, DimsOrder::NCHW, {1, 1, 32, 1}));
    model->attrs().set<int>("numOutputs", 1);

    auto pool = std::make_shared<ie::PoolingLayer>(ie::LayerParams{"pool", "Pooling", ie::Precision::FP16});

    pool->_kernel_x = inputW;
    pool->_kernel_y = inputH;
    pool->_stride_x = 1;
    pool->_stride_y = 1;
    pool->_type = ie::PoolingLayer::PoolType::AVG;

    frontEnd->parsePooling(model, pool, {input}, {output});

    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1);
    auto stages = model->getStages();
    ASSERT_EQ(stages.front()->type(), StageType::ReduceMean);
}