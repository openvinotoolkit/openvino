// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_UpliftReluTest : public GraphTransformerTest {
 protected:
    PassSet pipeline;
    Model model;
    const DimsOrder layout  = DimsOrder::CHW;
    const DataDesc dataDesc = {DataType::FP16, layout, {3, 5, 6}};
 public:
    void InitPipeline() {
        pipeline = PassSet();
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->upliftActivationStages());
        pipeline.addPass(passManager->dumpModel("upliftActivationStages"));
        pipeline.addPass(passManager->adjustDataLayout());
        pipeline.addPass(passManager->dumpModel("adjustDataLayout"));
        pipeline.addPass(passManager->adjustDataLocation());
        pipeline.addPass(passManager->dumpModel("adjustDataLocation"));
        pipeline.addPass(passManager->finalCheck());
    }

    Stage GetStage(size_t index) {
        auto stages = model->getStages();
        auto stageIter = stages.begin();
        std::advance(stageIter, index);
        return *stageIter;
    }

};
TEST_F(VPU_UpliftReluTest, CopyAndRelu) {
    InitCompileEnv();

    model = CreateModel();
    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output = model->addOutputData("Output", dataDesc);
    model->attrs().set<int>("numOutputs", 1);

    auto data1 = model->addNewData("data1", dataDesc);
    auto data2 = model->addNewData("data2", dataDesc);

    // sequence : input                  - Copy1 - data1 - Copy2 - data2 - ReLU - Output
    // optimized: input - ReLU - dataNew - Copy1 - data1 - Copy2 -              - Output

    stageBuilder->addCopyStage(model, "Copy1", nullptr,        input,  data1, "");
    stageBuilder->addCopyStage(model, "Copy2", nullptr,        data1,  data2, "");
    stageBuilder->addReLUStage(model, "ReLU",  nullptr, 0.01f, data2,  output);
    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 3U);
    ASSERT_EQ(GetStage(0)->type(), StageType::LeakyRelu);
}


TEST_F(VPU_UpliftReluTest, StopOnNonTransitive) {
    InitCompileEnv();

    model = CreateModel();
    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output1 = model->addOutputData("Output1", dataDesc);
    auto output2 = model->addOutputData("Output2", dataDesc);
    model->attrs().set<int>("numOutputs", 2);

    auto data1 = model->addNewData("data1", dataDesc);
    auto data2 = model->addNewData("data2", dataDesc);

    // sequence : input  - Copy1 - data1                   - Copy2 - data2 - ReLU - Output1
    //                                  \- CopyO - Output2
    // optimized: input - Copy1 - data1  - ReLU  - dataNew - Copy2 -              - Output1
    //                                  \- CopyO - Output2

    auto copy1stage = stageBuilder->addCopyStage(model, "Copy1", nullptr,     input,  data1  , "");
    auto copy2stage = stageBuilder->addCopyStage(model, "Copy2", nullptr,     data1,  data2  , "");
    auto copyOstage = stageBuilder->addCopyStage(model, "CopyO", nullptr,     data1,  output2, "");
    auto reluStage  = stageBuilder->addReLUStage(model, "ReLU",  nullptr, 0., data2,  output1);
    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 4U);
    ASSERT_EQ(GetStage(0)->type(), StageType::Copy);
    ASSERT_EQ(copy1stage->input(0)->name(), input->name());
    // Copy2 should have changed its input
    ASSERT_NE(copy2stage->input(0)->name(), data1->name());

    // inserted description should be identical
    ASSERT_EQ(copy2stage->input(0)->desc().toTensorDesc(), data1->desc().toTensorDesc());

    ASSERT_EQ(reluStage->input(0)->name(), data1->name());
    ASSERT_EQ(copyOstage->input(0)->name(), data1->name());
}
