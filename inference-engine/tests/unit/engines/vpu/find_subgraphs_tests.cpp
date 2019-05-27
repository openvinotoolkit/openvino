// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/containers.hpp>
#include <vpu/utils/range.hpp>

#include "graph_transformer_tests.hpp"

using namespace InferenceEngine;

class VPU_FindSubGraphsTest : public VPU_GraphTransformerTest {
protected:
    vpu::PassSet pipeline;
    vpu::Model::Ptr model;

public:
    void initSimpleModel() {
        InitCompileEnv();

        model = CreateModel();

        auto input = model->addInputData(
                "Input",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));
        model->attrs().set<int>("numInputs", 1);

        auto output = model->addOutputData(
                "Output1",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));
        model->attrs().set<int>("numOutputs", 1);

        auto data1 = model->addNewData(
                "data1",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));

        auto data2 = model->addNewData(
                "data2",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));

        auto data3 = model->addNewData(
                "data3",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));

        auto data4 = model->addNewData(
                "data4",
                vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));

        stageBuilder->addPowerStage(model, input->name(), nullptr, 0.0, 1.0, 0.0, input, data1);
        stageBuilder->addPowerStage(model, data1->name(), nullptr, 0.0, 1.0, 0.0, data1, data2);
        stageBuilder->addPowerStage(model, data2->name(), nullptr, 0.0, 1.0, 0.0, data2, data3);
        stageBuilder->addPowerStage(model, data3->name(), nullptr, 0.0, 1.0, 0.0, data3, data4);
        stageBuilder->addPowerStage(model, data4->name(), nullptr, 0.0, 1.0, 0.0, data4, output);

        pipeline.addPass(passManager->dumpModel("initial"));

        pipeline.addPass(passManager->findSubGraphs());
        pipeline.addPass(passManager->dumpModel("findSubGraphs"));
    }
};

TEST_F(VPU_FindSubGraphsTest, canCallfindSubGraphsPass) {
    config.numberOfNodesInOneSubGraph = 1;
    initSimpleModel();

    ASSERT_NO_THROW(pipeline.run(model));
}

TEST_F(VPU_FindSubGraphsTest, canMergeAllStagesInOneSubGraph) {
    config.numberOfNodesInOneSubGraph = 5;
    initSimpleModel();

    int maxSubGraphs = 1;

    ASSERT_NO_THROW(pipeline.run(model));

    auto curMaxSubGraphs = model->numberOfSubGraphs();
    ASSERT_EQ(curMaxSubGraphs, maxSubGraphs);
}

TEST_F(VPU_FindSubGraphsTest, canSplitGraphToTwoSubGraphs) {
    config.numberOfNodesInOneSubGraph = 3;
    initSimpleModel();

    ASSERT_NO_THROW(pipeline.run(model));

    auto curMaxSubGraphs = model->numberOfSubGraphs();
    ASSERT_EQ(curMaxSubGraphs, 2);

    for (int i = 0; i < curMaxSubGraphs; i++) {
        auto subGraph = vpu::toVector(model->getSubGraphStages(i));
        ASSERT_TRUE(subGraph.size() <= 3);

        for (const auto& stage : subGraph) {
            auto curSubGraph = stage->subGraphNumber();
            ASSERT_EQ(curSubGraph, i);
        }
    }
}

TEST_F(VPU_FindSubGraphsTest, canGetNextStagesWithCondition) {
    config.numberOfNodesInOneSubGraph = 3;
    initSimpleModel();

    ASSERT_NO_THROW(pipeline.run(model));

    auto curMaxSubGraphs = model->numberOfSubGraphs();
    ASSERT_EQ(curMaxSubGraphs, 2);

    auto subGraph0 = vpu::toVector(model->getSubGraphStages(0));
    auto stage0 = subGraph0[0];

    auto alwaysTrue = [](const vpu::Stage a) noexcept {
        return true;
    };
    auto res = vpu::toVector(stage0->nextStages(alwaysTrue));
    auto ref = vpu::toVector(stage0->nextStages());

    ASSERT_EQ(res.size(), ref.size());
    for (int i = 0; i < res.size(); i++) {
        ASSERT_EQ(res[i]->name(), ref[i]->name());
    }
}

TEST_F(VPU_FindSubGraphsTest, canGetPrevStagesWithCondition) {
    config.numberOfNodesInOneSubGraph = 3;
    initSimpleModel();

    ASSERT_NO_THROW(pipeline.run(model));

    auto curMaxSubGraphs = model->numberOfSubGraphs();
    ASSERT_EQ(curMaxSubGraphs, 2);

    auto subGraph1 = vpu::toVector(model->getSubGraphStages(1));
    auto stage1 = subGraph1[0];

    auto alwaysTrue = [](const vpu::Stage a)noexcept {
        return true;
    };
    auto res = vpu::toVector(stage1->prevStages(alwaysTrue));
    auto ref = vpu::toVector(stage1->prevStages());

    ASSERT_EQ(res.size(), ref.size());
    for (int i = 0; i < res.size(); i++) {
        ASSERT_EQ(res[i]->name(), ref[i]->name());
    }
}
