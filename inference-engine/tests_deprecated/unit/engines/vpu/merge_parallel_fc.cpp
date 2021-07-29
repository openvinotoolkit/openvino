// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include <vpu/stages/stub_stage.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <initializer_list>

using namespace vpu;

namespace {

using Dimensions = std::vector<std::size_t>;

struct FCDescriptor {
    Dimensions weights;
    Dimensions biases;
    Dimensions scales;
    Dimensions output;

    FCDescriptor(Dimensions new_weights, Dimensions new_biases, Dimensions new_scales, Dimensions new_output)
        : weights(std::move(new_weights))
        , biases(std::move(new_biases))
        , scales(std::move(new_scales))
        , output(std::move(new_output)) {}
};

class VPU_MergeParallelFCTestBase : public GraphTransformerTest, public testing::WithParamInterface<std::tuple<Dimensions, std::vector<FCDescriptor>>> {
public:
    void SetUp() override {
        GraphTransformerTest::SetUp();
        InitCompileEnv();
        InitPipeline();
    }

    void CreateModelWithParallelFC(const Dimensions& inputDimensions, const std::vector<FCDescriptor>& descriptors) {
        model = CreateModel();

        const auto input = model->addInputData("Input", CreateDescriptor(inputDimensions));
        model->attrs().set<int>("numInputs", 1);

        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            const auto& fcDims = descriptors[i];
            const auto& weights = model->addConstData("Weights#" + std::to_string(i), DataDesc{fcDims.weights});
            const auto& biases  = fcDims.biases.empty() ? model->addFakeData() : model->addConstData("Biases#" + std::to_string(i), DataDesc{fcDims.biases});
            const auto& scales  = fcDims.scales.empty() ? model->addFakeData() : model->addConstData("Scales#" + std::to_string(i), DataDesc{fcDims.scales});
            const auto& output  = model->addNewData("Output#" + std::to_string(i), CreateDescriptor(fcDims.output));
            model->addNewStage<StubStage>(
                "FC#" + std::to_string(i),
                StageType::StubFullyConnected,
                nullptr,
                {input, weights, biases, scales},
                {output});

            stageBuilder->addReLUStage(
                model,
                "ReLU#" + std::to_string(i),
                nullptr,
                0.0f,
                output,
                model->addOutputData("NetworkOutput#" + std::to_string(i), CreateDescriptor(fcDims.output)));
        }
    }

    void Compile() {
        pipeline.run(model);
    }

protected:
    static DataDesc CreateReferenceOutputDescriptor(const std::vector<FCDescriptor>& descriptors) {
        std::vector<DataDesc> outputDescriptors;
        std::transform(descriptors.begin(), descriptors.end(), std::back_inserter(outputDescriptors),
            [](const FCDescriptor& fcDescriptor) { return CreateDescriptor(fcDescriptor.output); });

        return mergeDescriptors(outputDescriptors);
    }

    static DataDesc CreateReferenceScalesDescriptor(const std::vector<FCDescriptor>& descriptors) {
        if (descriptors.front().scales.size() == 0) {
            return DataDesc{{1}};
        }

        std::vector<DataDesc> scalesDescriptors;
        std::transform(descriptors.begin(), descriptors.end(), std::back_inserter(scalesDescriptors),
            [](const FCDescriptor& fcDescriptor) { return CreateDescriptor(fcDescriptor.scales); });

        return mergeDescriptors(scalesDescriptors);
    }

    static DataDesc CreateReferenceBiasesDescriptor(const std::vector<FCDescriptor>& descriptors) {
        if (descriptors.front().biases.size() == 0) {
            return DataDesc{{1}};
        }

        std::vector<DataDesc> biasesDescriptors;
        std::transform(descriptors.begin(), descriptors.end(), std::back_inserter(biasesDescriptors),
            [](const FCDescriptor& fcDescriptor) { return CreateDescriptor(fcDescriptor.biases); });

        return mergeDescriptors(biasesDescriptors);
    }

    static DataDesc CreateReferenceWeightsDescriptor(const std::vector<FCDescriptor>& descriptors) {
        std::vector<DataDesc> weightsDescriptors;
        std::transform(descriptors.begin(), descriptors.end(), std::back_inserter(weightsDescriptors),
            [](const FCDescriptor& fcDescriptor) { return CreateDescriptor(fcDescriptor.weights); });

        return mergeDescriptors(weightsDescriptors);
    }

    static DataDesc mergeDescriptors(const std::vector<DataDesc>& descriptors) {
        auto merged = descriptors.front();
        merged.setDim(Dim::C, std::accumulate(descriptors.begin(), descriptors.end(), 0,
            [](int reduction, const DataDesc& descriptor) { return reduction + descriptor.dim(Dim::C); }));
        return merged;
    }

    static DataDesc CreateDescriptor(const Dimensions& dimensions) {
        return DataDesc{dimensions};
    }

    void InitPipeline() {
        pipeline = PassSet();
        pipeline.addPass(passManager->dumpModel("before-merge-parallel-fc"));
        pipeline.addPass(passManager->initialCheck());
        pipeline.addPass(passManager->mergeParallelFC());
        pipeline.addPass(passManager->dumpModel("after-merge-parallel-fc"));
    }

    PassSet pipeline;
    Model model;
};

class VPU_MergeParallelFCTest : public VPU_MergeParallelFCTestBase {
public:
    void AssertMergedCorrectly(const Dimensions& inputDimensions, const std::vector<FCDescriptor>& descriptors) const {
        if (descriptors.empty()) {
            return;
        }

        const auto& stages = model->getStages() | asVector();

        ASSERT_EQ(stages.size(), 2 + descriptors.size());
        const auto& fc = stages.front();
        const auto& split = stages[1];

        ASSERT_EQ(fc->type(), StageType::StubFullyConnected);
        ASSERT_EQ(split->type(), StageType::Split);

        ASSERT_TRUE(fc->numInputs() >= 2 && fc->numInputs() <= 4);

        ASSERT_EQ(fc->input(0)->desc(), CreateDescriptor(inputDimensions));
        ASSERT_EQ(fc->input(1)->desc(), CreateReferenceWeightsDescriptor(descriptors));
        ASSERT_EQ(fc->input(2)->desc(), CreateReferenceBiasesDescriptor(descriptors));
        ASSERT_EQ(fc->input(3)->desc(), CreateReferenceScalesDescriptor(descriptors));

        ASSERT_TRUE(fc->numOutputs() == 1);
        ASSERT_EQ(fc->output(0)->desc(), CreateReferenceOutputDescriptor(descriptors));

        ASSERT_EQ(split->numInputs(), 1);
        ASSERT_EQ(split->input(0)->desc(), fc->output(0)->desc());

        ASSERT_EQ(split->numOutputs(), descriptors.size());
        for (std::size_t i = 0; i < split->numOutputs(); ++i) {
            ASSERT_EQ(split->output(i)->desc(), CreateDescriptor(descriptors[i].output));
        }

        for (std::size_t i = 2; i < stages.size(); ++i) {
            const auto& relu  = stages[i];
            ASSERT_EQ(relu->type(), StageType::Relu);

            ASSERT_EQ(relu->numInputs(), 1);
            ASSERT_EQ(relu->numOutputs(), 1);

            ASSERT_EQ(split->output(i - 2)->desc(), relu->input(0)->desc());
            ASSERT_EQ(CreateDescriptor(descriptors[i - 2].output), relu->output(0)->desc());
        }
    }
};

class VPU_NoMergeParallelFCTest : public VPU_MergeParallelFCTestBase {
public:
    void AssertNotMergedCorrectly(const Dimensions& inputDimensions, const std::vector<FCDescriptor>& descriptors) {
        if (descriptors.empty()) {
            return;
        }

        const auto& stages = model->getStages() | asVector();
        ASSERT_EQ(stages.size(), 2 * descriptors.size());

        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            ASSERT_EQ(stages[2 * i + 0]->type(), StageType::StubFullyConnected);
            ASSERT_EQ(stages[2 * i + 1]->type(), StageType::Relu);
        }
    }
};

const std::initializer_list<Dimensions> inputDimensions = {
    {13, 21}
};

const std::initializer_list<std::vector<FCDescriptor>> FCDescriptorsMerge = {
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({5, 21}, {}, {}, {13, 5}),
    },
    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({5, 21}, {}, {21}, {13, 5}),
    },
    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({5, 21}, {21}, {}, {13, 5}),
    },
    {
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
        FCDescriptor({5, 21}, {21}, {21}, {13, 5}),
    },
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({5, 21}, {}, {}, {13, 5}),
        FCDescriptor({11, 21}, {}, {}, {13, 11}),
    },
    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({5, 21}, {}, {21}, {13, 5}),
        FCDescriptor({11, 21}, {}, {21}, {13, 11}),
    },
    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({5, 21}, {21}, {}, {13, 5}),
        FCDescriptor({11, 21}, {21}, {}, {13, 11}),
    },
    {
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
        FCDescriptor({5, 21}, {21}, {21}, {13, 5}),
        FCDescriptor({11, 21}, {21}, {21}, {13, 11}),
    }
};

const std::initializer_list<std::vector<FCDescriptor>> FCDescriptorsNoMerge {
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },

    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {}, {21}, {13, 7}),
    },
    {
        FCDescriptor({7, 21}, {21}, {}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
        FCDescriptor({7, 21}, {21}, {21}, {13, 7}),
    },
};

TEST_P(VPU_MergeParallelFCTest, MergeParallelFCMergeCases) {
    const auto& testParams          = GetParam();
    const auto& testInputDimensions = std::get<0>(testParams);
    const auto& testFCDescriptors   = std::get<1>(testParams);

    CreateModelWithParallelFC(testInputDimensions, testFCDescriptors);
    ASSERT_NO_THROW(Compile());

    AssertMergedCorrectly(testInputDimensions, testFCDescriptors);
}

TEST_P(VPU_NoMergeParallelFCTest, NoMergeParallelFCMergeCases) {
    const auto& testParams          = GetParam();
    const auto& testInputDimensions = std::get<0>(testParams);
    const auto& testFCDescriptors   = std::get<1>(testParams);

    CreateModelWithParallelFC(testInputDimensions, testFCDescriptors);
    ASSERT_NO_THROW(Compile());

    AssertNotMergedCorrectly(testInputDimensions, testFCDescriptors);
}

INSTANTIATE_TEST_SUITE_P(MergeParallelFCTest, VPU_MergeParallelFCTest, testing::Combine(
    testing::ValuesIn(inputDimensions),
    testing::ValuesIn(FCDescriptorsMerge)
));

INSTANTIATE_TEST_SUITE_P(NoMergeParallelFCTest, VPU_NoMergeParallelFCTest, testing::Combine(
    testing::ValuesIn(inputDimensions),
    testing::ValuesIn(FCDescriptorsNoMerge)
));

}

