// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/model/data_contents/replicated_data_content.hpp>

namespace vpu {

namespace {

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder) {
    }

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(decomposeSwish);

    for (const auto& swish : model->getStages()) {
        if (swish->type() != StageType::Swish) {
            continue;
        }
        const auto inputData = swish->input(0);
        const auto outputData = swish->output(0);
        const auto name = swish->name();
        const auto& layer = swish->origLayer();
        const auto beta = swish->attrs().get<float>("beta");

        model->removeStage(swish);
        auto sigmoidInput = inputData;

        if (beta != 1.0f) {
            const auto betaDesc = DataDesc(inputData->desc());
            const auto betaConst = model->addConstData(inputData->name() + "@beta", betaDesc,
                                                       replicateContent(beta, betaDesc.totalDimSize(), betaDesc));
            const auto prodOutput = model->addNewData(inputData->name() + "@prod-x-beta", inputData->desc());
            _stageBuilder->addProdStage(
                    model,
                    name + "@prod-x-beta",
                    layer,
                    inputData,
                    betaConst,
                    prodOutput);
            sigmoidInput = prodOutput;
        }
        const auto sigmoidDesc = inputData->desc();
        const auto sigmoidOutput = model->addNewData(inputData->name() + "@sigmoid", sigmoidDesc);

        _stageBuilder->addSigmoidStage(
                model,
                name + "@sigmoid",
                layer,
                {sigmoidInput},
                {sigmoidOutput});
        _stageBuilder->addProdStage(
                model,
                name + "@prod",
                layer,
                inputData,
                sigmoidOutput,
                outputData);
    }
}

}  // namespace

Pass::Ptr PassManager::decomposeSwish() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
