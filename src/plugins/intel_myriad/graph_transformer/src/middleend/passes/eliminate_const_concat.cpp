// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <blob_factory.hpp>

#include <vector>
#include <memory>

namespace vpu {

namespace {

class PassImpl final : public PerStagePass {
public:
    PassImpl() : PerStagePass({StageType::StubConcat}) {}

    void runForStage(const Model& model, const Stage& stage) override;
};

void PassImpl::runForStage(const Model& model, const Stage& stage) {
    VPU_PROFILE(eliminateConstConcat);

    //
    // Check for supported case
    //

    for (const auto& input : stage->inputs()) {
        if (input->usage() != DataUsage::Const) {
            return;
        }

        if (input->numConsumers() != 1) {
            return;
        }
    }

    //
    // Create merged blob
    //

    const auto& offsets = stage->attrs().get<std::vector<DimValues>>("offsets");

    const auto output = stage->output(0);

    const auto elemSize = output->desc().elemSize();
    const auto& outputDims = output->desc().dims();
    const auto outputDimsPerm = output->desc().dimsOrder().toPermutation();


    const auto generator = [&](const ie::Blob::Ptr& blob) {
        const auto mergedPtr = blob->buffer().as<uint8_t*>();
        IE_ASSERT(mergedPtr != nullptr);

        for (const auto& inputEdge : stage->inputEdges()) {
            const auto input = inputEdge->input();
            const auto& inputOffset = offsets.at(checked_cast<size_t>(inputEdge->portInd()));

            const auto& inputDims = input->desc().dims();
            const auto inputDimsPerm = input->desc().dimsOrder().toPermutation();

            const auto inputContent = input->content();
            const auto inputPtr = inputContent->get<uint8_t>();
            IE_ASSERT(inputPtr != nullptr);

            ie::parallel_for(input->desc().totalDimSize(), [&](int inputInd1D) {
                // Convert 1D index into ND
                DimValues inputIndND;
                auto tempInputInd1D = inputInd1D;
                for (auto dim : inputDimsPerm) {
                    const auto curDimSize = inputDims[dim];

                    const auto curDimInd = tempInputInd1D % curDimSize;
                    inputIndND.set(dim, curDimInd);

                    tempInputInd1D /= curDimSize;
                }

                DimValues outputIndND;
                for (const auto& p : inputIndND) {
                    const auto dim = p.first;
                    const auto inInd = p.second;
                    const auto outInd = inputOffset.get(dim, 0) + inInd;
                    outputIndND.set(dim, outInd);
                }

                // Convert ND index into 1D
                int outputInd1D = 0;
                int multiplier = 1;
                for (auto dim : outputDimsPerm) {
                    const auto curDimSize = outputDims[dim];
                    const auto curDimInd = outputIndND[dim];

                    outputInd1D += curDimInd * multiplier;
                    multiplier *= curDimSize;
                }

                std::copy_n(inputPtr + inputInd1D * elemSize, elemSize, mergedPtr + outputInd1D * elemSize);
            });
        }
    };

    //
    // Replace concat output with pre-calculated Data
    //

    const auto constOutput = model->addConstData(output->name(), output->desc(), generator);

    for (const auto& consumerEdge : output->consumerEdges()) {
        model->replaceStageInput(consumerEdge, constOutput);
    }

    model->removeStage(stage);
}

}  // namespace

Pass::Ptr PassManager::eliminateConstConcat() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
