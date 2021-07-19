// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/stages/mx_stage.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/private_plugin_config.hpp>

#include "graph_transformer_tests.hpp"

using namespace vpu;

using VPU_AdjustDataLocationTest = GraphTransformerTest;

//                                            -> [Data 2] -> (4/SW) -> [Output 1]
//                               -> (2/Split)
//                                            -> [Data 3] -> (5/SW) -> [Output 2]
// [Input] -> (1/HW) -> [Data 1]
//                                            -> [Data 4] -> (6/SW) -> [Output 3]
//                               -> (3/Split)
//                                            -> [Data 5] -> (7/SW) -> [Output 4]
//
// In order to allocate SHAVEs for SW Stages we need to move [Data 1] to DDR and redirect its consumers.
//

TEST_F(VPU_AdjustDataLocationTest, FlushCMX_TwoSpecialConsumers) {
    config.set(InferenceEngine::MYRIAD_NUMBER_OF_SHAVES, "1");
    config.set(InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES, "1");
    InitCompileEnv();

    DataDesc dataDesc1(DataType::FP16, DimsOrder::NCHW, {CMX_SLICE_SIZE / (2 * 2), 1, 2, 1});
    DataDesc dataDesc2(DataType::FP16, DimsOrder::NCHW, {CMX_SLICE_SIZE / (2 * 2), 1, 1, 1});

    auto model = CreateModel();

    auto input = model->addInputData("Input", dataDesc1);
    model->attrs().set<int>("numInputs", 1);

    auto output1 = model->addOutputData("Output 1", dataDesc2);
    auto output2 = model->addOutputData("Output 2", dataDesc2);
    auto output3 = model->addOutputData("Output 3", dataDesc2);
    auto output4 = model->addOutputData("Output 4", dataDesc2);
    model->attrs().set<int>("numOutputs", 4);

    auto data1 = model->addNewData("Data 1", dataDesc1);
    auto data2 = model->addNewData("Data 2", dataDesc2);
    auto data3 = model->addNewData("Data 3", dataDesc2);
    auto data4 = model->addNewData("Data 4", dataDesc2);
    auto data5 = model->addNewData("Data 5", dataDesc2);

    auto fake = model->addFakeData();

    auto hwStage = model->addNewStage<MyriadXHwStage>(
        "1/HW",
        StageType::MyriadXHwOp,
        nullptr,
        {input, fake, fake, fake},
        {data1});
    hwStage->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);

    stageBuilder->addSplitStage(model, "2/Split", nullptr, Dim::C, data1, {data2, data3});
    stageBuilder->addSplitStage(model, "3/Split", nullptr, Dim::C, data1, {data4, data5});

    stageBuilder->addSoftMaxStage(model, "4/SW", nullptr, data2, output1, Dim::W);
    stageBuilder->addSoftMaxStage(model, "5/SW", nullptr, data3, output2, Dim::W);
    stageBuilder->addSoftMaxStage(model, "6/SW", nullptr, data4, output3, Dim::W);
    stageBuilder->addSoftMaxStage(model, "7/SW", nullptr, data5, output4, Dim::W);

    PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"));
    pipeline.addPass(passManager->adjustDataLayout());
    pipeline.addPass(passManager->dumpModel("adjustDataLayout"));
    pipeline.addPass(passManager->processSpecialStages());
    pipeline.addPass(passManager->dumpModel("processSpecialStages"));
    pipeline.addPass(passManager->adjustDataLocation());
    pipeline.addPass(passManager->dumpModel("adjustDataLocation"));
    pipeline.addPass(passManager->finalCheck());

    pipeline.run(model);

    ASSERT_EQ(data1->dataLocation().location, Location::CMX);
    ASSERT_EQ(data1->numConsumers(), 1);

    auto data1Consumer = data1->singleConsumer();
    auto data1ConsumerOutput = data1Consumer->output(0);
    ASSERT_EQ(data1Consumer->type(), StageType::Copy);
    ASSERT_EQ(data1ConsumerOutput->dataLocation().location, Location::BSS);
    ASSERT_EQ(data1ConsumerOutput->numChildDatas(), 4);
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataToDataEdges(), [data2](const DataToDataAllocation& e) { return e->child() == data2; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataToDataEdges(), [data3](const DataToDataAllocation& e) { return e->child() == data3; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataToDataEdges(), [data4](const DataToDataAllocation& e) { return e->child() == data4; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataToDataEdges(), [data5](const DataToDataAllocation& e) { return e->child() == data5; }));
}

//
//                               -> (HW 2) -> [Data 2] ->
// [Input] -> (HW 1) -> [Data 1]                          (Sum) -> [Output]
//                                                     ->
//

TEST_F(VPU_AdjustDataLocationTest, SpillWithBranch) {
    InitCompileEnv();

    const auto& env = CompileEnv::get();

    const auto maxCmxSizeBytes = env.resources.numCMXSlices * CMX_SLICE_SIZE;
    const auto maxCmxSizeElems = maxCmxSizeBytes / sizeof(fp16_t);
    const auto testSizeElems = alignVal<int>((2 * maxCmxSizeElems) / 3, HW_STRIDE_ALIGNMENT / sizeof(fp16_t));

    DataDesc dataDesc(DataType::FP16, DimsOrder::NCHW, {testSizeElems, 1, 1, 1});

    auto model = CreateModel();

    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output = model->addOutputData("Output", dataDesc);
    model->attrs().set<int>("numOutputs", 1);

    auto data1 = model->addNewData("Data 1", dataDesc);
    auto data2 = model->addNewData("Data 2", dataDesc);

    auto fake = model->addFakeData();

    auto hw1 = model->addNewStage<MyriadXHwStage>(
        "HW 1",
        StageType::MyriadXHwOp,
        nullptr,
        {input, fake, fake, fake},
        {data1});
    hw1->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);

    // Create Sum first, so it will be the first consumer of "Data 1"
    auto sum = stageBuilder->addSumStage(model, "Sum", nullptr, data1, data2, output);

    auto hw2 = model->addNewStage<MyriadXHwStage>(
        "HW 2",
        StageType::MyriadXHwOp,
        nullptr,
        {data1, fake, fake, fake},
        {data2});
    hw2->attrs().set<HwOpType>("hwOpType", HwOpType::POOL);

    PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"), "dump");
    pipeline.addPass(passManager->adjustDataLayout(), "adjustDataLayout");
    pipeline.addPass(passManager->dumpModel("adjustDataLayout"), "dump");
    pipeline.addPass(passManager->adjustDataLocation(), "adjustDataLocation");
    pipeline.addPass(passManager->dumpModel("adjustDataLocation"), "dump");

    pipeline.run(model);

    auto hw1Output = hw1->output(0);
    ASSERT_EQ(hw1Output->dataLocation().location, Location::CMX);

    auto copyStage = hw1Output->singleConsumer();
    ASSERT_EQ(copyStage->type(), StageType::Copy);

    auto copyStageOutput = copyStage->output(0);
    ASSERT_EQ(copyStageOutput->dataLocation().location, Location::BSS);

    ASSERT_EQ(copyStageOutput->numConsumers(), 2);
    for (const auto& copyStageOutputConsumer : copyStageOutput->consumers()) {
        ASSERT_TRUE(copyStageOutputConsumer == hw2 || copyStageOutputConsumer == sum);
    }
}
