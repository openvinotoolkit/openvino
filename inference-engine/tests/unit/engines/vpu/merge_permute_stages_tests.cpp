// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_MergePermuteTest : public GraphTransformerTest {
 protected:
    PassSet pipeline;
    Model model;
    using PermuteDims = PermutationDimsMap;
 public:
    void SetUp() override {
        GraphTransformerTest::SetUp();
        InitCompileEnv();
    }

    void CreateModelWithTwoPermutes(const DimsOrder& layout, const PermuteDims& firstPermute, const PermuteDims& secondPermute) {
        model = CreateModel();

        const DataDesc dataDesc(DataType::FP16, layout, MakeStubDims(layout));
        auto input = model->addInputData("Input", dataDesc);
        model->attrs().set<int>("numInputs", 1);

        auto output = model->addOutputData("Output", dataDesc);
        model->attrs().set<int>("numOutputs", 1);

        auto data1 = model->addNewData("data1", dataDesc);

        stageBuilder->addPermuteStage(model,
                                      "Permute1",
                                      nullptr,
                                      input,
                                      data1,
                                      firstPermute
                                    );

        stageBuilder->addPermuteStage(model,
                                      "Permute2",
                                      nullptr,
                                      data1,
                                      output,
                                      secondPermute
                                    );
        InitPipeline();
    }

    void CreateModelWithThreePermutes(const DimsOrder& layout, const PermuteDims& firstPermute, const PermuteDims& secondPermute, const PermuteDims& thirdPermute) {
        model = CreateModel();

        const DataDesc dataDesc(DataType::FP16, layout, MakeStubDims(layout));
        auto input = model->addInputData("Input", dataDesc);
        model->attrs().set<int>("numInputs", 1);

        auto output = model->addOutputData("Output", dataDesc);
        model->attrs().set<int>("numOutputs", 1);

        auto data1 = model->addNewData("data1", dataDesc);
        auto data2 = model->addNewData("data2", dataDesc);

        stageBuilder->addPermuteStage(model,
                                      "Permute1",
                                      nullptr,
                                      input,
                                      data1,
                                      firstPermute
                                    );

        stageBuilder->addPermuteStage(model,
                                      "Permute2",
                                      nullptr,
                                      data1,
                                      data2,
                                      secondPermute
                                    );

        stageBuilder->addPermuteStage(model,
                                      "Permute3",
                                      nullptr,
                                      data2,
                                      output,
                                      thirdPermute
                                    );
        InitPipeline();
    }

    void CreateModelWithPermuteAndReorder(const DimsOrder& layoutIn, const DimsOrder& layoutOut, const PermuteDims& firstPermute) {
        model = CreateModel();

        const DataDesc dataDescIn (DataType::FP16, layoutIn,  MakeStubDims(layoutIn));
        const DataDesc dataDescOut(DataType::FP16, layoutOut, MakeStubDims(layoutIn));  // That's not a typo: we use same Dim-to-size mapping for in and out.
        auto input = model->addInputData("Input", dataDescIn);
        model->attrs().set<int>("numInputs", 1);

        auto output = model->addOutputData("Output", dataDescOut);
        model->attrs().set<int>("numOutputs", 1);

        auto data1 = model->addNewData("data1", dataDescIn);

        stageBuilder->addPermuteStage(model,
                                      "Permute",
                                      nullptr,
                                      input,
                                      data1,
                                      firstPermute
                                    );

        stageBuilder->addReorderStage(model,
                                      "Reorder",
                                      nullptr,
                                      data1,
                                      output
                                    );
        InitPipeline();
    }

    void InitPipeline() {
        pipeline = PassSet();
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->adjustDataLayout());
        pipeline.addPass(passManager->dumpModel("adjustDataLayout"));
        pipeline.addPass(passManager->mergePermuteStages());
        pipeline.addPass(passManager->dumpModel("mergePermuteStages"));
        pipeline.addPass(passManager->adjustDataLocation());
        pipeline.addPass(passManager->dumpModel("adjustDataLocation"));
        pipeline.addPass(passManager->finalCheck());
    }

    // Create fake dimension size array, using dimension count from layout
    DimValues MakeStubDims(const DimsOrder& layout)
    {
        const auto perm = layout.toPermutation();
        const int numDms = layout.numDims();
        DimValues dims;
        for (int ind = 0; ind < numDms; ++ind) {
            dims.set(perm[ind], 2 + ind * 2); // 2, 4, 6 ...
        }
        return dims;
    }

    const PermuteDims& GetPermuteDimsForStage(size_t index) {
        auto stageIter = model->getStages().begin();
        std::advance(stageIter, index);
        auto permuteStage = *stageIter;

        IE_ASSERT(permuteStage->type() == StageType::Permute);
        return permuteStage->attrs().get<PermutationDimsMap>("permutation");
    }

};
TEST_F(VPU_MergePermuteTest, InternalFunctions) {
    // Meaning of integer permutation vectors:
    // suppose you have vector {1, 2, 0}, expand it in mapping, where key would be vector index:
    // [0] = 1
    // [1] = 2
    // [2] = 0
    // The key is destination dimension, and the value is the source dimension.
    // For CHW, C==2, H==1, W==0 index, so permutation above would be equal to:
    // W <- H
    // H <- C
    // C <- W

    const auto order = DimsOrder::NCDHW;
    auto origMapping = PermuteDims{{Dim::N, Dim::C}, {Dim::C, Dim::D}, {Dim::D, Dim::H}, {Dim::H, Dim::W}, {Dim::W, Dim::N}};
    auto permVector = permuteMapToVector(origMapping, order, order);
    {
        ASSERT_EQ(permVector, (PermutationIndexVector{4, 0, 1, 2, 3}));
        auto identityCheckMapping = permuteVectorToMap(permVector, order, order);
        ASSERT_EQ(origMapping, identityCheckMapping);
    }

    {
        auto permVectorCombined = combinePermutationVectors(permVector, permVector);
        ASSERT_EQ(permVectorCombined, (PermutationIndexVector{3, 4, 0, 1, 2}));
        auto combinedMapping = permuteVectorToMap(permVectorCombined, order, order);
        auto identityCombinedMapping = PermuteDims{{Dim::N, Dim::D}, {Dim::C, Dim::H}, {Dim::D, Dim::W}, {Dim::H, Dim::N}, {Dim::W, Dim::C}};
        ASSERT_EQ(combinedMapping, identityCombinedMapping);

    }
    {
        auto permVectorCombined = combinePermutationVectors(PermutationIndexVector{0, 1, 2, 4, 3}, PermutationIndexVector{0, 1, 3, 2, 4});
        ASSERT_EQ(permVectorCombined, (PermutationIndexVector{0, 1, 4, 2, 3}));
    }

    {
        auto permVector1 = permuteMapToVector(PermuteDims{{Dim::H, Dim::C}, {Dim::W, Dim::H}, {Dim::C, Dim::W}},
                                              DimsOrder::CHW, DimsOrder::CHW);
        ASSERT_EQ(permVector1, (PermutationIndexVector{1, 2, 0}));
        auto permVector2 = permuteMapToVector(PermuteDims{{Dim::H, Dim::C}, {Dim::C, Dim::H}, {Dim::W, Dim::W}},
                                              DimsOrder::CHW, DimsOrder::CHW);
        ASSERT_EQ(permVector2, (PermutationIndexVector{0, 2, 1}));

        auto permVectorCombined = combinePermutationVectors(permVector1, permVector2);
        ASSERT_EQ(permVectorCombined, (PermutationIndexVector{1, 0, 2}));
    }
}

TEST_F(VPU_MergePermuteTest, TwoPermutes) {
    // TLDR: {H, C} + {W, H} => {W, C}.
    // How does permutation should combine?
    // according to stages/permute.cpp, parsePermute() stage, in  PermutationDimsMap first is destination dimension,
    // and the second value in pair - is source dimension.
    // for {Dim::W, Dim::H}, data in W output will be relevant to H dimension.
    // If we have {Dim::H, Dim::C} in first permute, and {Dim::W, Dim::H} in the second, the resulting permutation is
    // {Dim::W, Dim::C}  - basically, we taking the dest (first in pair) from the second permute,
    // and the source (second in pair) value  from the first permute.

    // clockwise permutes on 4 dims.
    CreateModelWithTwoPermutes(DimsOrder::NCHW,
                               PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::N}, {Dim::N, Dim::W}},
                               PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::N}, {Dim::N, Dim::W}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(GetPermuteDimsForStage(0), (PermuteDims{{Dim::W, Dim::C}, {Dim::H, Dim::N}, {Dim::C, Dim::W}, {Dim::N, Dim::H}}));

    // counter-clockwise permutes on 5 dims
    CreateModelWithTwoPermutes(DimsOrder::NCDHW,
                               PermuteDims{{Dim::N, Dim::C}, {Dim::C, Dim::D}, {Dim::D, Dim::H}, {Dim::H, Dim::W}, {Dim::W, Dim::N}},
                               PermuteDims{{Dim::N, Dim::C}, {Dim::C, Dim::D}, {Dim::D, Dim::H}, {Dim::H, Dim::W}, {Dim::W, Dim::N}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(GetPermuteDimsForStage(0), (PermuteDims{{Dim::N, Dim::D}, {Dim::C, Dim::H}, {Dim::D, Dim::W}, {Dim::H, Dim::N}, {Dim::W, Dim::C}}));

    // partial compensation (3 dim)
    CreateModelWithTwoPermutes(DimsOrder::CHW,
                               PermuteDims{{Dim::H, Dim::C}, {Dim::W, Dim::H}, {Dim::C, Dim::W}},
                               PermuteDims{{Dim::C, Dim::H}, {Dim::W, Dim::W}, {Dim::H, Dim::C}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(GetPermuteDimsForStage(0), (PermuteDims{{Dim::H, Dim::W}, {Dim::W, Dim::H}, {Dim::C, Dim::C}}));

    // two compensating permutes (3 dim)
    CreateModelWithTwoPermutes(DimsOrder::CHW,
                               PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}},
                               PermuteDims{{Dim::H, Dim::W}, {Dim::C, Dim::H}, {Dim::W, Dim::C}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(model->getStages().front()->type(), StageType::Copy);
}

TEST_F(VPU_MergePermuteTest, ThreePermutes) {
    // three compensating permutes (3 dim)
    CreateModelWithThreePermutes(DimsOrder::CHW,
                                 PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}},
                                 PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}},
                                 PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(model->getStages().front()->type(), StageType::Copy);

    // three merging permutes (5 dim)
    CreateModelWithThreePermutes(DimsOrder::NCDHW,
                                 PermuteDims{{Dim::N, Dim::C}, {Dim::C, Dim::N}, {Dim::D, Dim::D}, {Dim::H, Dim::H}, {Dim::W, Dim::W}},
                                 PermuteDims{{Dim::N, Dim::N}, {Dim::C, Dim::D}, {Dim::D, Dim::C}, {Dim::H, Dim::H}, {Dim::W, Dim::W}},
                                 PermuteDims{{Dim::N, Dim::N}, {Dim::C, Dim::C}, {Dim::D, Dim::H}, {Dim::H, Dim::D}, {Dim::W, Dim::W}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(GetPermuteDimsForStage(0), (PermuteDims{{Dim::H, Dim::N}, {Dim::C, Dim::D}, {Dim::N, Dim::C}, {Dim::D, Dim::H}, {Dim::W, Dim::W}}));
}

TEST_F(VPU_MergePermuteTest, TwoSequences) {
    model = CreateModel();
    const auto layout = DimsOrder::CHW;
    const DataDesc dataDesc(DataType::FP16, layout, MakeStubDims(layout));
    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output = model->addOutputData("Output", dataDesc);
    model->attrs().set<int>("numOutputs", 1);

    // sequence: input - Permute11 - data1 - Permute12 - data2 - Copy - data3 - Permute21 - data4 - Permute22 - output

    auto data1 = model->addNewData("data1", dataDesc);
    auto data2 = model->addNewData("data2", dataDesc);
    auto data3 = model->addNewData("data3", dataDesc);
    auto data4 = model->addNewData("data4", dataDesc);

    stageBuilder->addPermuteStage(model,
                                  "Permute11",
                                  nullptr,
                                  input,
                                  data1,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    stageBuilder->addPermuteStage(model,
                                  "Permute12",
                                  nullptr,
                                  data1,
                                  data2,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );

    stageBuilder->addCopyStage(model,
                               "Copy",
                               nullptr,
                               data2,
                               data3,
                               ""
                               );

    stageBuilder->addPermuteStage(model,
                                  "Permute21",
                                  nullptr,
                                  data3,
                                  data4,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    stageBuilder->addPermuteStage(model,
                                  "Permute22",
                                  nullptr,
                                  data4,
                                  output,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 3U); // 2 Permute and 1 copy stage.
}

TEST_F(VPU_MergePermuteTest, OutputBeteweenPermutes) {
    model = CreateModel();
    const auto layout = DimsOrder::CHW;
    const DataDesc dataDesc(DataType::FP16, layout, MakeStubDims(layout));
    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output1 = model->addOutputData("Output1", dataDesc);
    auto output2 = model->addOutputData("Output2", dataDesc);
    model->attrs().set<int>("numOutputs", 2);

    // sequence: input - Permute1 - Output1 - Permute2 - Output2

    stageBuilder->addPermuteStage(model,
                                  "Permute1",
                                  nullptr,
                                  input,
                                  output1,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    stageBuilder->addPermuteStage(model,
                                  "Permute2",
                                  nullptr,
                                  output1,
                                  output2,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 2U);
}

TEST_F(VPU_MergePermuteTest, DataSourceSplitBeteweenPermutes) {
    model = CreateModel();
    const auto layout = DimsOrder::CHW;
    const DataDesc dataDesc(DataType::FP16, layout, MakeStubDims(layout));
    auto input = model->addInputData("Input", dataDesc);
    model->attrs().set<int>("numInputs", 1);

    auto output1 = model->addOutputData("Output1", dataDesc);
    auto output2 = model->addOutputData("Output2", dataDesc);
    model->attrs().set<int>("numOutputs", 2);

    auto data1 = model->addNewData("data1", dataDesc);

    // sequence: input - Permute1 - data1 - Permute2 - Output1
    //                                    \ Copy     - Output2

    stageBuilder->addPermuteStage(model,
                                  "Permute1",
                                  nullptr,
                                  input,
                                  data1,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    stageBuilder->addPermuteStage(model,
                                  "Permute2",
                                  nullptr,
                                  data1,
                                  output1,
                                  PermuteDims{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::W}}
                                );
    stageBuilder->addCopyStage(model, "Copy", nullptr, data1, output2, "");
    InitPipeline();

    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 3U);
}

TEST_F(VPU_MergePermuteTest, PermuteAndReorder) {
    CreateModelWithPermuteAndReorder(DimsOrder::CHW,
                                     DimsOrder::HWC,
                                     PermuteDims{{Dim::W, Dim::W}, {Dim::H, Dim::C}, {Dim::C, Dim::H}});
    ASSERT_NO_THROW(pipeline.run(model));
    ASSERT_EQ(model->getStages().size(), 1U);
    ASSERT_EQ(GetPermuteDimsForStage(0), (PermuteDims{{Dim::W, Dim::W}, {Dim::H, Dim::C}, {Dim::C, Dim::H}}));
}
