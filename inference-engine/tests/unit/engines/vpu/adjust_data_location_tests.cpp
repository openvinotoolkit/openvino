// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/allocator.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/numeric.hpp>

#include "graph_transformer_tests.hpp"

using VPU_AdjustDataLocationTest = VPU_GraphTransformerTest;

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
    config.numSHAVEs = 1;
    config.numCMXSlices = 1;
    InitCompileEnv();

    vpu::DataDesc dataDesc1(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {vpu::CMX_SLICE_SIZE / (2 * 2), 1, 2, 1});
    vpu::DataDesc dataDesc2(vpu::DataType::FP16, vpu::DimsOrder::NCHW, {vpu::CMX_SLICE_SIZE / (2 * 2), 1, 1, 1});

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

    auto hwStage = model->addNewStage<vpu::MyriadXHwStage>(
        "1/HW",
        vpu::StageType::MyriadXHwOp,
        nullptr,
        {input, fake, fake, fake},
        {data1});
    hwStage->attrs().set<vpu::HwOpType>("hwOpType", vpu::HwOpType::POOL);

    stageBuilder->addSplitStage(model, "2/Split", nullptr, vpu::Dim::C, data1, {data2, data3});
    stageBuilder->addSplitStage(model, "3/Split", nullptr, vpu::Dim::C, data1, {data4, data5});

    stageBuilder->addSoftMaxStage(model, "4/SW", nullptr, data2, output1, vpu::Dim::W);
    stageBuilder->addSoftMaxStage(model, "5/SW", nullptr, data3, output2, vpu::Dim::W);
    stageBuilder->addSoftMaxStage(model, "6/SW", nullptr, data4, output3, vpu::Dim::W);
    stageBuilder->addSoftMaxStage(model, "7/SW", nullptr, data5, output4, vpu::Dim::W);

    vpu::PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"));
    pipeline.addPass(passManager->adjustDataLayout());
    pipeline.addPass(passManager->dumpModel("adjustDataLayout"));
    pipeline.addPass(passManager->processSpecialStages());
    pipeline.addPass(passManager->dumpModel("processSpecialStages"));
    pipeline.addPass(passManager->adjustDataLocation());
    pipeline.addPass(passManager->dumpModel("adjustDataLocation"));
    pipeline.addPass(passManager->finalCheck());

    pipeline.run(model);

    ASSERT_EQ(data1->location(), vpu::DataLocation::CMX);
    ASSERT_EQ(data1->numConsumers(), 1);

    auto data1Consumer = data1->singleConsumer();
    auto data1ConsumerOutput = data1Consumer->output(0);
    ASSERT_EQ(data1Consumer->type(), vpu::StageType::Copy);
    ASSERT_EQ(data1ConsumerOutput->location(), vpu::DataLocation::BSS);
    ASSERT_EQ(data1ConsumerOutput->numChildDatas(), 4);
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataEdges(), [data2](const vpu::SharedAllocation& e) { return e->child() == data2; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataEdges(), [data3](const vpu::SharedAllocation& e) { return e->child() == data3; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataEdges(), [data4](const vpu::SharedAllocation& e) { return e->child() == data4; }));
    ASSERT_TRUE(contains(data1ConsumerOutput->childDataEdges(), [data5](const vpu::SharedAllocation& e) { return e->child() == data5; }));
}
