// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include <vpu/model/data_contents/replicated_data_content.hpp>

#include <precision_utils.h>

using namespace vpu;

using VPU_EliminateConstConcatTest = GraphTransformerTest;

TEST_F(VPU_EliminateConstConcatTest, EliminateCase_1D) {
    InitCompileEnv();

    const DataDesc dataDesc1(DataType::FP16, DimsOrder::C, {8});
    const DataDesc dataDesc2(DataType::FP16, DimsOrder::C, {4});

    const DataDesc dataDescConcat(DataType::FP16, DimsOrder::C, {dataDesc1.dim(Dim::C) + dataDesc2.dim(Dim::C)});

    const auto model = CreateModel();

    const auto constData1 = model->addConstData("const1", dataDesc1, replicateContent(1.0f, dataDesc1.totalDimSize(), dataDesc1));
    const auto constData2 = model->addConstData("const2", dataDesc2, replicateContent(2.0f, dataDesc2.totalDimSize(), dataDesc2));

    const auto concatData = model->addNewData("concat", dataDescConcat);

    const auto input = model->addInputData("input", dataDescConcat);
    const auto output = model->addOutputData("output", dataDescConcat);

    model->attrs().set<int>("numInputs", 1);
    model->attrs().set<int>("numOutputs", 1);

    stageBuilder->addConcatStage(model, "concat", nullptr, Dim::C, {constData1, constData2}, concatData);
    stageBuilder->addNoneStage(model, "none", nullptr, {input, concatData}, {output});

    PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"));
    pipeline.addPass(passManager->eliminateConstConcat());
    pipeline.addPass(passManager->dumpModel("eliminateConstConcat"));

    pipeline.run(model);

    ASSERT_EQ(model->numStages(), 1);

    const auto noneStage = model->getStages().front();
    ASSERT_EQ(noneStage->type(), StageType::None);

    ASSERT_EQ(noneStage->numInputs(), 2);
    ASSERT_EQ(noneStage->input(0), input);

    const auto mergedConcatData = noneStage->input(1);
    ASSERT_EQ(mergedConcatData->usage(), DataUsage::Const);
    ASSERT_EQ(mergedConcatData->desc().dimsOrder(), dataDescConcat.dimsOrder());
    ASSERT_EQ(mergedConcatData->desc().dims(), dataDescConcat.dims());

    const auto mergedContent = mergedConcatData->content();
    const auto mergedContentPtr = mergedContent->get<fp16_t>();

    for (int ind = 0; ind < dataDesc1.dim(Dim::C); ++ind) {
        ASSERT_EQ(mergedContentPtr[ind], ie::PrecisionUtils::f32tof16(1.0f)) << ind;
    }
    for (int ind = 0; ind < dataDesc2.dim(Dim::C); ++ind) {
        ASSERT_EQ(mergedContentPtr[ind + dataDesc1.dim(Dim::C)], ie::PrecisionUtils::f32tof16(2.0f)) << ind;
    }
}

TEST_F(VPU_EliminateConstConcatTest, EliminateCase_2D) {
    InitCompileEnv();

    const DataDesc dataDesc1(DataType::FP16, DimsOrder::NC, {4, 4});
    const DataDesc dataDesc2(DataType::FP16, DimsOrder::NC, {2, 4});

    const DataDesc dataDescConcat(DataType::FP16, DimsOrder::NC, {dataDesc1.dim(Dim::C) + dataDesc2.dim(Dim::C), dataDesc1.dim(Dim::N)});

    const auto model = CreateModel();

    const auto constData1 = model->addConstData("const1", dataDesc1, replicateContent(1.0f, dataDesc1.totalDimSize(), dataDesc1));
    const auto constData2 = model->addConstData("const2", dataDesc2, replicateContent(2.0f, dataDesc2.totalDimSize(), dataDesc2));

    const auto concatData = model->addNewData("concat", dataDescConcat);

    const auto input = model->addInputData("input", dataDescConcat);
    const auto output = model->addOutputData("output", dataDescConcat);

    model->attrs().set<int>("numInputs", 1);
    model->attrs().set<int>("numOutputs", 1);

    stageBuilder->addConcatStage(model, "concat", nullptr, Dim::C, {constData1, constData2}, concatData);
    stageBuilder->addNoneStage(model, "none", nullptr, {input, concatData}, {output});

    PassSet pipeline;
    pipeline.addPass(passManager->dumpModel("initial"));
    pipeline.addPass(passManager->eliminateConstConcat());
    pipeline.addPass(passManager->dumpModel("eliminateConstConcat"));

    pipeline.run(model);

    ASSERT_EQ(model->numStages(), 1);

    const auto noneStage = model->getStages().front();
    ASSERT_EQ(noneStage->type(), StageType::None);

    ASSERT_EQ(noneStage->numInputs(), 2);
    ASSERT_EQ(noneStage->input(0), input);

    const auto mergedConcatData = noneStage->input(1);
    ASSERT_EQ(mergedConcatData->usage(), DataUsage::Const);
    ASSERT_EQ(mergedConcatData->desc().dimsOrder(), dataDescConcat.dimsOrder());
    ASSERT_EQ(mergedConcatData->desc().dims(), dataDescConcat.dims());

    const auto mergedContent = mergedConcatData->content();
    const auto mergedContentPtr = mergedContent->get<fp16_t>();

    for (int indN = 0; indN < dataDescConcat.dim(Dim::N); ++indN) {
        for (int indC = 0; indC < dataDesc1.dim(Dim::C); ++indC) {
            const auto mergedContentInd = mergedConcatData->elemOffset(DimValues{{Dim::C, indC}, {Dim::N, indN}}) / sizeof(fp16_t);
            ASSERT_EQ(mergedContentPtr[mergedContentInd], ie::PrecisionUtils::f32tof16(1.0f)) << indC;
        }
        for (int indC = 0; indC < dataDesc2.dim(Dim::C); ++indC) {
            const auto mergedContentInd = mergedConcatData->elemOffset(DimValues{{Dim::C, dataDesc1.dim(Dim::C) + indC}, {Dim::N, indN}}) / sizeof(fp16_t);
            ASSERT_EQ(mergedContentPtr[mergedContentInd], ie::PrecisionUtils::f32tof16(2.0f)) << indC;
        }
    }
}
