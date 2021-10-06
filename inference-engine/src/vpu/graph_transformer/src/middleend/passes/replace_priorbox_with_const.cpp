// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/stages/stub_stage.hpp>
#include <vpu/model/data_contents/priorbox_contents.hpp>

#include <precision_utils.h>

#include <cmath>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <queue>

namespace vpu {

namespace {

//
// UnusedDataRemover class deletes data that has no consumers,
// and also recursively deletes all its unused predecessors, including
// those stage that don't produce any data as a result of previous data removing.
//

class UnusedDataRemover {
public:
    explicit UnusedDataRemover(const vpu::Model& model) : _model(model) {}

    void run(const Data& start_data);

private:
    const vpu::Model& _model;
    std::queue<Data> _datas;
};

void UnusedDataRemover::run(const Data& start_data) {
    _datas.push(start_data);

    while (!_datas.empty()) {
        const auto& data = _datas.front();
        const auto& producer = data->producer();

        if (producer != nullptr && producer->nextStages().empty()) {
            for (const auto &inEdge : producer->inputEdges()) {
                _datas.push(inEdge->input());
            }

            _model->removeStage(producer);
        }

        if (data->numConsumers() == 0) {
            _model->removeUnusedData(data);
        }

        _datas.pop();
    }
}

//
// ReplacePriorBoxWithConst pass removes all StubPriorBox and StubPriorBoxClustered,
// calculates a const PriorBox data and removes all henceforth unused input data.
//

class PassImpl final : public Pass {
public:
    explicit PassImpl(StageBuilder::Ptr stageBuilder) : _stageBuilder(std::move(stageBuilder)) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const vpu::Model &model) {
    VPU_PROFILE(replacePriorBoxWithConst);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubPriorBox && stage->type() != StageType::StubPriorBoxClustered) {
            continue;
        }
        IE_ASSERT(stage->numInputs() == 2);
        IE_ASSERT(stage->numOutputs() == 1);

        const auto& layer = stage->origLayer();
        const auto& input0 = stage->input(0);
        const auto& input1 = stage->input(1);
        const auto& output = stage->output(0);

        model->disconnectStage(stage);

        DataContent::Ptr content = nullptr;
        if (stage->type() == StageType::StubPriorBox) {
            content = std::make_shared<PriorBoxContent>(input0->desc(), input1->desc(), output->desc(), layer);
        } else {
            content = std::make_shared<PriorBoxClusteredContent>(input0->desc(), input1->desc(), output->desc(), layer);
        }

        auto resultData = model->addConstData(
            formatString("%s@const", output->name()),
            output->desc(),
            content);

        if (output->usage() == DataUsage::Output) {
            _stageBuilder->addCopyStage(
                model,
                formatString("%s@copy-output", stage->name()),
                layer,
                resultData,
                output,
                "replacePriorBoxWithConst");
        } else {
            IE_ASSERT(output->usage() == DataUsage::Intermediate);
            IE_ASSERT(output->numConsumers() > 0);

            for (const auto& consumer_edge : output->consumerEdges()) {
                model->replaceStageInput(consumer_edge, resultData);
            }
        }

        UnusedDataRemover remover(model);
        remover.run(input0);
        remover.run(input1);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::replacePriorBoxWithConst() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
