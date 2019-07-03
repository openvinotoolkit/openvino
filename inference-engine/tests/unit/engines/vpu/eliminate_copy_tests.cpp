// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/utility.hpp>

#include "graph_transformer_tests.hpp"

using VPU_EliminateCopyTest = VPU_GraphTransformerTest;

TEST_F(VPU_EliminateCopyTest, OneInputTwoConcats) {
    InitCompileEnv();

    auto model = CreateModel();

    auto input = model->addInputData(
        "Input",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));
    model->attrs().set<int>("numInputs", 1);

    auto output1 = model->addOutputData(
        "Output1",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 4, 1}));
    auto output2 = model->addOutputData(
        "Output2",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 4, 1}));
    model->attrs().set<int>("numOutputs", 2);

    auto outputCopy1 = model->duplicateData(output1, "copy");
    auto outputCopy2 = model->duplicateData(output2, "copy");
    stageBuilder->addCopyStage(model, outputCopy1->name(), nullptr, outputCopy1, output1);
    stageBuilder->addCopyStage(model, outputCopy2->name(), nullptr, outputCopy2, output2);

    auto data1 = model->addNewData(
        "data1",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 3, 1}));

    auto fake = model->addFakeData();

    auto hwStage = model->addNewStage<vpu::MyriadXHwStage>(
        "HW",
        vpu::StageType::MyriadXHwOp,
        nullptr,
        {input, fake, fake, fake},
        {data1});
    hwStage->attrs().set<vpu::HwOpType>("hwOpType", vpu::HwOpType::POOL);

    auto data2 = model->addNewData(
        "data2",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 1, 1}));
    auto data3 = model->addNewData(
        "data3",
        vpu::DataDesc(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {16, 16, 1, 1}));

    stageBuilder->addConcatStage(
        model,
        "Concat1",
        nullptr,
        vpu::Dim::C,
        {data1, data2},
        outputCopy1);
    stageBuilder->addConcatStage(
        model,
        "Concat2",
        nullptr,
        vpu::Dim::C,
        {data1, data3},
        outputCopy2);

    vpu::PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"));
    pipeline.addPass(passManager->adjustDataLayout());
    pipeline.addPass(passManager->dumpModel("adjustDataLayout"));
    pipeline.addPass(passManager->processSpecialStages());
    pipeline.addPass(passManager->dumpModel("processSpecialStages"));
    pipeline.addPass(passManager->adjustDataLocation());
    pipeline.addPass(passManager->dumpModel("adjustDataLocation"));
    pipeline.addPass(passManager->eliminateCopyStages());
    pipeline.addPass(passManager->dumpModel("eliminateCopyStages"));
    pipeline.addPass(passManager->finalCheck());

    pipeline.run(model);

    const auto& hwOutput = hwStage->output(0);
    ASSERT_NE(hwOutput->parentDataEdge(), nullptr);
    ASSERT_EQ(hwOutput->parentData(), outputCopy1);

    ASSERT_EQ(hwOutput->numConsumers(), 2);
    ASSERT_TRUE(contains(hwOutput->consumers(), [](const vpu::Stage& stage) { return stage->type() == vpu::StageType::Concat; }));
    ASSERT_TRUE(contains(hwOutput->consumers(), [](const vpu::Stage& stage) { return stage->type() == vpu::StageType::Copy; }));
}
