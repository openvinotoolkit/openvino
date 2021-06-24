// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/stages/stub_stage.hpp>
#include <vpu/model/data_contents/merge_fc_content.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace vpu {

namespace {

DataDesc mergeDescriptors(const DataVector& dataObjects) {
    const auto& targetDim = Dim::C;
    auto mergedDescriptor = dataObjects.front()->desc();
    const auto mergedDimension = std::accumulate(dataObjects.begin(), dataObjects.end(), 0,
        [targetDim](int reduction, const Data& data) { return reduction + data->desc().dims()[targetDim]; });
    mergedDescriptor.setDim(targetDim, mergedDimension);
    return mergedDescriptor;
}

Data mergeConstDataObjects(const Model& model, const DataVector& dataObjects) {
    if (dataObjects.empty()) {
        return model->addFakeData();
    }

    std::vector<DataContent::CPtr> contents;
    std::vector<DataDesc> descs;
    for (const auto& data : dataObjects) {
        contents.push_back(data->content());
        descs.push_back(data->desc());
    }

    auto mergedDesc = mergeDescriptors(dataObjects);

    auto content = std::make_shared<MergeFullyConnectedContentsByChannels>(contents, descs, mergedDesc);
    return model->duplicateData(dataObjects.front(), "@merge-parallel-fc", mergedDesc, content);
}

Data mergeOutputs(const Model& model, const DataVector& dataObjects) {
    return model->duplicateData(dataObjects.front(), "@merge-parallel-fc", mergeDescriptors(dataObjects));
}

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr new_builder) : builder(std::move(new_builder)) {}

    void run(const Model& model) override {
        for (const auto& data : model->datas()) {
            const auto& targets = data->consumers();

            if (targets.size() < 2) {
                continue;
            }

            auto isFullyConnected = [](const Stage& stage) { return stage->type() == StageType::StubFullyConnected; };
            if (!std::all_of(targets.begin(), targets.end(), isFullyConnected)) {
                continue;
            }

            const auto& front = targets.front();
            auto areInputsConsistent = [&front](const Stage& stage) {
                const bool areBiasesConsistent = front->input(2)->usage() == stage->input(2)->usage();
                const bool areScalesConsistent = front->input(3)->usage() == stage->input(3)->usage();
                return areBiasesConsistent && areScalesConsistent;
            };

            if (!std::all_of(targets.begin(), targets.end(), areInputsConsistent)) {
                // If there is a stage that has different number of real input objects (e.g. doesn't have biases const data object as input) merge cannot
                // be performed since we don't have a data to merge with. Option to fill by zero seems to be unreasonable in performance context.
                continue;
            }

            merge(model, data, targets | asVector());
        }
    }

private:
    void merge(const Model& model, const Data& input, const StageVector& targets) {
        DataVector weights, biases, scales, outputs;
        for (const auto& target : targets) {
            weights.push_back(target->input(1));

            if (target->input(2)->usage() != DataUsage::Fake) {
                biases.push_back(target->input(2));
            }

            if (target->input(3)->usage() != DataUsage::Fake) {
                scales.push_back(target->input(3));
            }

            outputs.push_back(target->output(0));
            model->disconnectStage(target);
        }

        const auto mergedBiases = mergeConstDataObjects(model, biases);
        const auto mergedWeights = mergeConstDataObjects(model, weights);
        const auto mergedScales = mergeConstDataObjects(model, scales);
        const auto mergedOutput = mergeOutputs(model, outputs);

        auto mergedStage = model->addNewStage<StubStage>(
            "merged-fc",
            targets.front()->type(),

            // vpu::StageNode assumes single original layer
            // in case of multiple stages collision may occur
            // there is no rule of choosing one layer among the rest
            // let original layer be nullptr in sake of consistency
            nullptr,

            {input, mergedWeights, mergedBiases, mergedScales},
            {mergedOutput});

        mergedStage->attrs() = targets.front()->attrs();

        builder->addSplitStage(
            model,
            mergedStage->name() + "@split",
            nullptr,
            Dim::C,
            mergedOutput,
            outputs);

        for (const auto& target : targets) {
            model->removeStage(target);
        }
    }

    StageBuilder::Ptr builder;
};

}  // namespace

Pass::Ptr PassManager::mergeParallelFC() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
