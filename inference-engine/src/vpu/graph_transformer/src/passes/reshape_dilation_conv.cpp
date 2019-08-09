// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <tuple>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <cmath>
#include <list>
#include <set>
#include <unordered_map>
#include <memory>

#include <vpu/stub_stage.hpp>
#include <vpu/sw/utility.hpp>
#include <vpu/compile_env.hpp>

#include <vpu/utils/handle.hpp>

namespace vpu {

namespace {

using ReplicatedDataMap = std::unordered_map<int, Data>;

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) :
            _stageBuilder(stageBuilder) {
    }

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(reshapeDilationConv);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubConv) {
            continue;
        }

        auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        auto dilationX = stage->attrs().get<int>("dilationX");
        auto dilationY = stage->attrs().get<int>("dilationY");
        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");

        if (dilationX <= 1 && dilationY <= 1) {
            continue;
        }

        if (groupSize != 1) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((padTop != padBottom) || (padLeft != padRight)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((dilationX != dilationY) || (dilationX != 2)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((kernelStrideX != 1) || (kernelStrideY != 1)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto output = stage->output(0);
        auto input_org = input;

        if (input->desc().dim(Dim::N) > 1) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((input->desc().dimsOrder() != DimsOrder::NCHW)
                || (output->desc().dimsOrder() != DimsOrder::NCHW)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        bool Expand_mark = false;
        // TODO
        const bool Use_pixel_alignment = false;
        int pixel_stride_alignment = STRIDE_ALIGNMENT
                / input->desc().elemSize();
        int InputExtended_width = input->desc().dim(Dim::W);
        int InputExtended_height = input->desc().dim(Dim::H);

        if (Use_pixel_alignment) {
            InputExtended_width = divUp(input->desc().dim(Dim::W),
                    dilationX * pixel_stride_alignment) * dilationX
                    * pixel_stride_alignment;
            InputExtended_height = divUp(input->desc().dim(Dim::H),
                    dilationY * pixel_stride_alignment) * dilationY
                    * pixel_stride_alignment;
        } else if ((divUp(input->desc().dim(Dim::W), dilationX)
                < pixel_stride_alignment)
                || (divUp(input->desc().dim(Dim::H), dilationY)
                        < pixel_stride_alignment)) {
            InputExtended_width = pixel_stride_alignment * dilationX;
            InputExtended_height = pixel_stride_alignment * dilationY;
        } else {
            InputExtended_width = divUp(input->desc().dim(Dim::W), dilationX)
                    * dilationX;
            InputExtended_height = divUp(input->desc().dim(Dim::H), dilationY)
                    * dilationY;
        }

        if ((((InputExtended_width % pixel_stride_alignment) == 0) && (InputExtended_width % (dilationX * pixel_stride_alignment) != 0))
                || (((InputExtended_height % pixel_stride_alignment) == 0) && (InputExtended_height % (dilationX * dilationY) != 0))) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        float InputExtended_scale = std::max(
                static_cast<float>(InputExtended_width)
                        / static_cast<float>(input->desc().dim(Dim::W)),
                static_cast<float>(InputExtended_height)
                        / static_cast<float>(input->desc().dim(Dim::H)));

        const float MAX_INPUTEXTENDED_SCALE = 1.8;
        const float MIN_INPUTEXTENDED_SCALE = 1;

        if (InputExtended_scale  >= MAX_INPUTEXTENDED_SCALE) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        Expand_mark = (InputExtended_scale > MIN_INPUTEXTENDED_SCALE);

        model->disconnectStageDatas(stage);

        // Expand input if need
        auto newDesc_input = input->desc();
        auto InputExtended = input;

        if (Expand_mark) {
            newDesc_input.setDim(Dim::W, InputExtended_width);
            newDesc_input.setDim(Dim::H, InputExtended_height);

            InputExtended = model->duplicateData(input, "@extended-input",
                    newDesc_input);

            _stageBuilder->addBroadcastStage(model,
                    stage->name() + "@expand-input", stage->origLayer(), input,
                    InputExtended);
        }

        DataDesc Reinterpret_inputdataDesc(DataType::FP16, DimsOrder::NCHW,
                { dilationX, InputExtended->desc().dim(Dim::W) / dilationX,
                        InputExtended->desc().dim(Dim::H),
                        InputExtended->desc().dim(Dim::C) });

        Data Reinterpret_inputdata;
        Reinterpret_inputdata = model->duplicateData(InputExtended,
                "@reinterpret-input-data", Reinterpret_inputdataDesc);

        auto reshape_stage = _stageBuilder->addReshapeStage(model,
                stage->name() + "@copy-reinterpret-input-data",
                stage->origLayer(), InputExtended, Reinterpret_inputdata);

        DataDesc Permuted_inputdataDesc(DataType::FP16, DimsOrder::NCHW,
                { InputExtended->desc().dim(Dim::W) / dilationX,
                        InputExtended->desc().dim(Dim::H),
                        InputExtended->desc().dim(Dim::C),
                        dilationX });

        Data Permuted_inputdata;
        Permuted_inputdata = model->duplicateData(InputExtended,
                "@permuted-input-data", Permuted_inputdataDesc);

        SmallVector<int, MAX_DIMS_64> ieOrder(4, -1);

        ieOrder[0] = 3;
        ieOrder[1] = 0;
        ieOrder[2] = 1;
        ieOrder[3] = 2;

        _stageBuilder->addPermuteStage(model,
                stage->origLayerName() + "@permute-input-data",
                stage->origLayer(), { Reinterpret_inputdata }, {
                        Permuted_inputdata }, ieOrder);

        // for conv output of subtensors
        auto padx_new = padLeft - (kernelSizeX - 1) * (dilationX - 1) / 2;
        auto pady_new = padTop - (kernelSizeY - 1) * (dilationY - 1) / 2;

        auto newDesc_Permuted_input = InputExtended->desc();
        newDesc_Permuted_input.setDim(Dim::W,
                (((InputExtended->desc().dim(Dim::W) + 2 * padx_new
                        - kernelSizeX) / kernelStrideX) + 1) / dilationX);
        newDesc_Permuted_input.setDim(Dim::H,
                ((InputExtended->desc().dim(Dim::H) + 2 * pady_new
                        - kernelSizeY) / kernelStrideY) + 1);
        newDesc_Permuted_input.setDim(Dim::C, output->desc().dim(Dim::C));
        newDesc_Permuted_input.setDim(Dim::N, dilationX);

        auto Subtensors_outputdata = model->duplicateData(output,
                "@SubTensors-OutputData", newDesc_Permuted_input);

        // for skip rows, use reshape n c h w/2 -> n c h/2 w
        auto Reshape_Permuted_inputdata_Desc = Permuted_inputdata->desc();
        Reshape_Permuted_inputdata_Desc.setDim(Dim::H,
                Permuted_inputdata->desc().dim(Dim::H) / dilationY);
        Reshape_Permuted_inputdata_Desc.setDim(Dim::W,
                Permuted_inputdata->desc().dim(Dim::W) * dilationY);
        auto Reshape_Permuted_inputdata = model->duplicateData(
                Permuted_inputdata, "@Reshape-Permuted-inputdata",
                Reshape_Permuted_inputdata_Desc);

        _stageBuilder->addReshapeStage(model,
                stage->name() + "@Reshape-Permuted-inputdata",
                stage->origLayer(), Permuted_inputdata,
                Reshape_Permuted_inputdata);

        auto Reshape_Permuted_outputdata_Desc = Subtensors_outputdata->desc();
        Reshape_Permuted_outputdata_Desc.setDim(Dim::H,
                Subtensors_outputdata->desc().dim(Dim::H) / dilationY);
        Reshape_Permuted_outputdata_Desc.setDim(Dim::W,
                Subtensors_outputdata->desc().dim(Dim::W) * dilationY);
        auto Reshape_Permuted_outputdata = model->duplicateData(
                Subtensors_outputdata, "@Reshape-Permuted-outputdata",
                Reshape_Permuted_outputdata_Desc);

        // Desc of sub input tensor
        DataDesc Sub_inputdataDesc(
                { Permuted_inputdata->desc().dim(Dim::W),
                        Permuted_inputdata->desc().dim(Dim::H) / dilationY,
                        Permuted_inputdata->desc().dim(Dim::C),
                        1 });

        Sub_inputdataDesc.reorder(DimsOrder::NCHW);

        // Desc of sub output tensor
        auto Sub_outputdataDesc = Subtensors_outputdata->desc();

        Sub_outputdataDesc.setDim(Dim::N, 1);
        Sub_outputdataDesc.setDim(Dim::C,
                Subtensors_outputdata->desc().dim(Dim::C));
        Sub_outputdataDesc.setDim(Dim::H,
                ((Sub_inputdataDesc.dim(Dim::H) + 2 * pady_new - kernelSizeY)
                        / kernelStrideY) + 1);
        Sub_outputdataDesc.setDim(Dim::W,
                ((Sub_inputdataDesc.dim(Dim::W) + 2 * padx_new - kernelSizeX)
                        / kernelStrideX) + 1);

        DataVector V_Sub_inputdata;
        std::vector<DimValues> V_Sub_inputdatasOffsets;

        DataVector V_Sub_outputdata;
        std::vector<DimValues> V_Sub_outputdatasOffsets;

        DataVector V_newWeights;
        DataVector V_newbiases;

        V_Sub_inputdata.reserve(dilationX * dilationY);
        V_Sub_inputdatasOffsets.reserve(dilationX * dilationY);
        V_Sub_outputdata.reserve(dilationX * dilationY);
        V_Sub_outputdatasOffsets.reserve(dilationX * dilationY);

        V_newWeights.reserve(dilationX * dilationY);
        V_newbiases.reserve(dilationX * dilationY);

        for (int dilationXInd = 0; dilationXInd < dilationX; ++dilationXInd) {
            for (int dilationYInd = 0; dilationYInd < dilationY;
                    ++dilationYInd) {
                Data Sub_inputdata;
                Sub_inputdata = model->duplicateData(Permuted_inputdata,
                        "@Sub-InputData", Sub_inputdataDesc);

                DimValues Sub_inputdatasOffsets;
                Sub_inputdatasOffsets.set(Dim::N, dilationXInd);
                Sub_inputdatasOffsets.set(Dim::W,
                        dilationYInd * Sub_inputdataDesc.dim(Dim::W));

                Data Sub_outputdata;
                Sub_outputdata = model->duplicateData(Subtensors_outputdata,
                        "@Sub_OutputData", Sub_outputdataDesc);

                DimValues Sub_outputdatasOffsets;
                Sub_outputdatasOffsets.set(Dim::N, dilationXInd);
                Sub_outputdatasOffsets.set(Dim::W,
                        dilationYInd * Sub_outputdataDesc.dim(Dim::W));

                // reuse weights and biases
                auto newWeights = model->duplicateData(weights, "@NewWeights",
                        weights->desc());
                auto newbiases = model->duplicateData(biases, "@Newbiases",
                        biases->desc());

                V_Sub_inputdata.emplace_back(Sub_inputdata);
                V_Sub_inputdatasOffsets.emplace_back(Sub_inputdatasOffsets);
                V_Sub_outputdata.emplace_back(Sub_outputdata);
                V_Sub_outputdatasOffsets.emplace_back(Sub_outputdatasOffsets);

                V_newWeights.emplace_back(newWeights);
                V_newbiases.emplace_back(newbiases);
            }
        }

        auto SplitPermutedInputDataStage = _stageBuilder->addSplitStage(model,
                stage->name() + "@Split-Permuted-InputData", stage->origLayer(),
                std::move(V_Sub_inputdatasOffsets), Reshape_Permuted_inputdata,
                V_Sub_inputdata);

        // sub tensors convolution
        for (int index = 0; index < dilationX * dilationY; ++index) {
            // Add SubDataConv stage
            auto newStage = model->addNewStage<StubStage>(
                    stage->origLayerName() + "@SubDataConv",
                    StageType::StubConv, stage->origLayer(), {
                            V_Sub_inputdata[index], V_newWeights[index],
                            V_newbiases[index] }, { V_Sub_outputdata[index] });

            newStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            newStage->attrs().set<int>("kernelSizeY", kernelSizeY);
            newStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            newStage->attrs().set<int>("kernelStrideY", kernelStrideY);
            newStage->attrs().set<int>("padLeft", padx_new);
            newStage->attrs().set<int>("padRight", padx_new);
            newStage->attrs().set<int>("padTop", pady_new);
            newStage->attrs().set<int>("padBottom", pady_new);
            newStage->attrs().set<int>("dilationX", 1);
            newStage->attrs().set<int>("dilationY", 1);
            newStage->attrs().set<int>("groupSize", groupSize);
            newStage->attrs().set<bool>("tryHW", true);
        }

        auto ConcatSubOutputDataStage = _stageBuilder->addConcatStage(model,
                stage->name() + "@Concat-Sub-OutputData", stage->origLayer(),
                std::move(V_Sub_outputdatasOffsets), V_Sub_outputdata,
                Reshape_Permuted_outputdata);

        _stageBuilder->addReshapeStage(model,
                stage->name() + "@Reshape-conv-outputdata", stage->origLayer(),
                Reshape_Permuted_outputdata, Subtensors_outputdata);

        // output permute
        DataDesc permute_outputdataDesc(DataType::FP16, DimsOrder::NCHW,
                { Subtensors_outputdata->desc().dim(Dim::C),
                        Subtensors_outputdata->desc().dim(Dim::H),
                        Subtensors_outputdata->desc().dim(Dim::W),
                        Subtensors_outputdata->desc().dim(Dim::N) });

        permute_outputdataDesc.setDim(Dim::N,
                Subtensors_outputdata->desc().dim(Dim::C));
        permute_outputdataDesc.setDim(Dim::C,
                Subtensors_outputdata->desc().dim(Dim::H));
        permute_outputdataDesc.setDim(Dim::H,
                Subtensors_outputdata->desc().dim(Dim::W));
        permute_outputdataDesc.setDim(Dim::W,
                Subtensors_outputdata->desc().dim(Dim::N));

        auto permute_outputdata = model->duplicateData(Subtensors_outputdata,
                "@Permuted-OutputData", permute_outputdataDesc);

        SmallVector<int, MAX_DIMS_64> ieOrder2(4, -1);

        ieOrder2[0] = 1;
        ieOrder2[1] = 2;
        ieOrder2[2] = 3;
        ieOrder2[3] = 0;

        _stageBuilder->addPermuteStage(model,
                stage->origLayerName() + "@Permute-OutputData",
                stage->origLayer(), { Subtensors_outputdata }, {
                        permute_outputdata }, ieOrder2);

        // Expand output if need
        if (Expand_mark) {
            auto Reinterpret_outputdataDesc = permute_outputdataDesc;
            Reinterpret_outputdataDesc.reorder(DimsOrder::NCHW);

            Reinterpret_outputdataDesc.setDim(Dim::C,
                    permute_outputdata->desc().dim(Dim::N));
            Reinterpret_outputdataDesc.setDim(Dim::H,
                    permute_outputdata->desc().dim(Dim::C));
            Reinterpret_outputdataDesc.setDim(Dim::W,
                    permute_outputdata->desc().dim(Dim::W)
                            * permute_outputdata->desc().dim(Dim::H));
            Reinterpret_outputdataDesc.setDim(Dim::N, 1);

            auto Reinterpret_outputdata = model->duplicateData(
                    permute_outputdata, "@Reinterpret-OutputData",
                    Reinterpret_outputdataDesc);

            _stageBuilder->addReshapeStage(model,
                    stage->name() + "@copy-Permute-OutputData",
                    stage->origLayer(), permute_outputdata,
                    Reinterpret_outputdata);

            auto ShrinkToOutputDataStage = _stageBuilder->addShrinkStage(model,
                    stage->name() + "@shrink-to-OutputData", stage->origLayer(),
                    Reinterpret_outputdata, output);

        } else {
            _stageBuilder->addReshapeStage(model,
                    stage->name() + "@copy-Permute-OutputData",
                    stage->origLayer(), permute_outputdata, output);
        }

        model->removeStage(stage);
    }
}
}  // namespace

Pass::Ptr PassManager::reshapeDilationConv() {
    return std::make_shared < PassImpl > (_stageBuilder);
}

}  // namespace vpu
