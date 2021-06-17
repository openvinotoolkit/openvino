// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

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

#include <vpu/stages/stub_stage.hpp>
#include <vpu/middleend/sw/utility.hpp>
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

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
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

        auto scaleFactor = stage->attrs().getOrDefault<float>("scaleFactor", 1.0f);

        IE_ASSERT(dilationX >= 1);
        IE_ASSERT(dilationY >= 1);
        if (dilationX <= 1 && dilationY <= 1) {
            continue;
        }

        if (groupSize != 1) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if (((padLeft % dilationX) !=0) || ((padTop % dilationY) !=0)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((std::max(dilationX, kernelStrideX) % std::min(dilationX, kernelStrideX)) ||
                (std::max(dilationY, kernelStrideY) % std::min(dilationY, kernelStrideY))) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto scales = stage->input(3);
        auto output = stage->output(0);

        if (input->desc().dim(Dim::N) > 1) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        if ((input->desc().dimsOrder() != DimsOrder::NCHW)
                || (output->desc().dimsOrder() != DimsOrder::NCHW)) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        const bool Use_pixel_alignment = true;
        int pixel_stride_alignment = HW_STRIDE_ALIGNMENT
                / input->desc().elemSize();
        int InputExtended_width = input->desc().dim(Dim::W);
        int InputExtended_height = input->desc().dim(Dim::H);

        if (Use_pixel_alignment) {
            InputExtended_width = divUp(input->desc().dim(Dim::W),
                    dilationX * pixel_stride_alignment) * dilationX
                    * pixel_stride_alignment;

            InputExtended_height = divUp(input->desc().dim(Dim::H), dilationY)
                    * dilationY;
        } else {
            InputExtended_width = divUp(input->desc().dim(Dim::W), dilationX)
                    * dilationX;
            InputExtended_height = divUp(input->desc().dim(Dim::H), dilationY)
                    * dilationY;
        }

        float InputExtended_scale = std::max(
                static_cast<float>(InputExtended_width)
                        / static_cast<float>(input->desc().dim(Dim::W)),
                static_cast<float>(InputExtended_height)
                        / static_cast<float>(input->desc().dim(Dim::H)));

        const float MAX_INPUTEXTENDED_SCALE = 1.8f;
        const float MIN_INPUTEXTENDED_SCALE = 1;

        if (InputExtended_scale >= MAX_INPUTEXTENDED_SCALE) {
            stage->attrs().set<bool>("tryHW", false);
            continue;
        }

        bool Expand_mark = false;

        Expand_mark = (InputExtended_scale > MIN_INPUTEXTENDED_SCALE);

        model->disconnectStage(stage);

        // Expand input if need
        auto newDesc_input = input->desc();
        auto InputExtended = input;

        if (Expand_mark) {
            newDesc_input.setDim(Dim::W, InputExtended_width);
            newDesc_input.setDim(Dim::H, InputExtended_height);

            InputExtended = model->duplicateData(input, "@extended-input",
                    newDesc_input);

            _stageBuilder->addPadStage(model, stage->name() + "@padding",
                    stage->origNode(),
                    PadMode::Constant, 0.0f, DimValues(),
                    DimValues({ { Dim::W, (InputExtended_width - input->desc().dim(Dim::W)) },
                    { Dim::H, (InputExtended_height - input->desc().dim(Dim::H)) }, }),
                    input,
                    InputExtended);
        }

        DataDesc Reinterpret_inputdataDesc(
            DataType::FP16,
            DimsOrder::NCHW,
            {
                dilationX,
                InputExtended->desc().dim(Dim::W) / dilationX,
                InputExtended->desc().dim(Dim::H),
                InputExtended->desc().dim(Dim::C)
            });

        Data Reinterpret_inputdata;
        Reinterpret_inputdata = model->duplicateData(InputExtended,
                "@reinterpret-input-data", Reinterpret_inputdataDesc);

        auto reshape_stage = _stageBuilder->addReshapeStage(model,
                stage->name() + "@copy-reinterpret-input-data",
                stage->origNode(), InputExtended, Reinterpret_inputdata);

        DataDesc Permuted_inputdataDesc(
            DataType::FP16,
            DimsOrder::NCHW,
            {
                InputExtended->desc().dim(Dim::W) / dilationX,
                InputExtended->desc().dim(Dim::H),
                InputExtended->desc().dim(Dim::C),
                dilationX
            });

        Data Permuted_inputdata;
        Permuted_inputdata = model->duplicateData(InputExtended,
                "@permuted-input-data", Permuted_inputdataDesc);

        _stageBuilder->addPermuteStage(
            model,
            stage->origLayerName() + "@permute-input-data",
            stage->origNode(),
            Reinterpret_inputdata,
            Permuted_inputdata,
            DimValues_<Dim>{{Dim::W, Dim::H}, {Dim::H, Dim::C}, {Dim::C, Dim::N}, {Dim::N, Dim::W}});

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
                stage->origNode(), Permuted_inputdata,
                Reshape_Permuted_inputdata);

        // Desc of sub input tensor
        DataDesc Sub_inputdataDesc(
                { Permuted_inputdata->desc().dim(Dim::W),
                        Permuted_inputdata->desc().dim(Dim::H) / dilationY,
                        Permuted_inputdata->desc().dim(Dim::C), 1 });

        Sub_inputdataDesc.reorder(DimsOrder::NCHW);

        auto Sub_output_dilationX_dimenion = (dilationX / kernelStrideX) > 1 ? (dilationX / kernelStrideX) : 1;
        auto Sub_output_dilationY_dimenion = (dilationY / kernelStrideY) > 1 ? (dilationY / kernelStrideY) : 1;
        auto kernelStrideX_new = (dilationX / kernelStrideX) > 1 ? 1 : (kernelStrideX / dilationX);
        auto kernelStrideY_new = (dilationY / kernelStrideY) > 1 ? 1 : (kernelStrideY / dilationY);

        // for conv output of subtensors
        auto padLeft_new = padLeft / dilationX;
        auto padRight_new = padRight / dilationX;
        auto padTop_new =  padTop / dilationY;
        auto padBottom_new = padBottom / dilationY;

        auto Subtensors_outputdataDesc = InputExtended->desc();

        Subtensors_outputdataDesc.setDim(Dim::W,
                ((InputExtended->desc().dim(Dim::W) + padLeft + padRight
                        - dilationX * (kernelSizeX - 1) - 1 + kernelStrideX)
                        / kernelStrideX) / Sub_output_dilationX_dimenion);

        Subtensors_outputdataDesc.setDim(Dim::H,
                (InputExtended->desc().dim(Dim::H) + padTop + padBottom
                        - dilationY * (kernelSizeY - 1) - 1 + kernelStrideY)
                        / kernelStrideY);

        Subtensors_outputdataDesc.setDim(Dim::C, output->desc().dim(Dim::C));
        Subtensors_outputdataDesc.setDim(Dim::N, Sub_output_dilationX_dimenion);

        auto Subtensors_outputdata = model->duplicateData(output,
                "@SubTensors-OutputData", Subtensors_outputdataDesc);

        // Desc of sub output tensor
        auto Real_sub_outputdataDesc = Subtensors_outputdata->desc();

        int Real_sub_outputdata_width = ((Sub_inputdataDesc.dim(Dim::W)
                + padLeft_new + padRight_new - kernelSizeX) / kernelStrideX_new)
                + 1;
        int Real_sub_outputdata_height = ((Sub_inputdataDesc.dim(Dim::H)
                + padTop_new + padBottom_new - kernelSizeY) / kernelStrideY_new)
                + 1;

        if (Real_sub_outputdata_width != Subtensors_outputdataDesc.dim(Dim::W)) {
            padRight_new = (Subtensors_outputdataDesc.dim(Dim::W) - 1) * kernelStrideX_new
                    + kernelSizeX - padLeft_new - Sub_inputdataDesc.dim(Dim::W);
            Real_sub_outputdata_width = Subtensors_outputdataDesc.dim(Dim::W);
        }

        if (Real_sub_outputdata_height != (Subtensors_outputdataDesc.dim(Dim::H) / Sub_output_dilationY_dimenion)) {
            padBottom_new = (Subtensors_outputdataDesc.dim(Dim::H) - 1) * kernelStrideY_new
                    + kernelSizeY - padTop_new - Sub_inputdataDesc.dim(Dim::H);
            Real_sub_outputdata_height = (Subtensors_outputdataDesc.dim(Dim::H) / Sub_output_dilationY_dimenion);
        }

        bool Sub_outputdata_expand = false;
        int Sub_outputdata_width = Real_sub_outputdata_width;

        if ((Real_sub_outputdata_width % pixel_stride_alignment) != 0) {
            Sub_outputdata_expand = true;
            Sub_outputdata_width = divUp(Real_sub_outputdata_width,
                    pixel_stride_alignment) * pixel_stride_alignment;
            padRight_new = (Sub_outputdata_width - 1) * kernelStrideX_new
                    + kernelSizeX - padLeft_new - Sub_inputdataDesc.dim(Dim::W);
        }

        Real_sub_outputdataDesc.setDim(Dim::N, 1);
        Real_sub_outputdataDesc.setDim(Dim::C,
                Subtensors_outputdata->desc().dim(Dim::C));

        Real_sub_outputdataDesc.setDim(Dim::H, Real_sub_outputdata_height);
        Real_sub_outputdataDesc.setDim(Dim::W, Real_sub_outputdata_width);

        DataVector V_Sub_inputdata;
        std::vector<DimValues> V_Sub_inputdatasOffsets;

        DataVector V_Sub_outputdata;
        std::vector<DimValues> V_Sub_outputdatasOffsets;

        V_Sub_inputdata.reserve(Sub_output_dilationX_dimenion * Sub_output_dilationY_dimenion);
        V_Sub_inputdatasOffsets.reserve(Sub_output_dilationX_dimenion * Sub_output_dilationY_dimenion);

        V_Sub_outputdata.reserve(Sub_output_dilationX_dimenion * Sub_output_dilationY_dimenion);
        V_Sub_outputdatasOffsets.reserve(Sub_output_dilationX_dimenion * Sub_output_dilationY_dimenion);

        for (int dilationXInd = 0; dilationXInd < dilationX; dilationXInd += (dilationX / Sub_output_dilationX_dimenion)) {
            for (int dilationYInd = 0; dilationYInd < dilationY;
                    dilationYInd += (dilationY / Sub_output_dilationY_dimenion)) {
                Data Sub_inputdata;
                Sub_inputdata = model->duplicateData(Permuted_inputdata,
                        "@Sub-InputData", Sub_inputdataDesc);

                DimValues Sub_inputdatasOffsets;
                Sub_inputdatasOffsets.set(Dim::N, dilationXInd);
                Sub_inputdatasOffsets.set(Dim::W,
                        dilationYInd * Sub_inputdataDesc.dim(Dim::W));

                V_Sub_inputdata.emplace_back(Sub_inputdata);
                V_Sub_inputdatasOffsets.emplace_back(Sub_inputdatasOffsets);

                Data Real_sub_outputdata;
                Real_sub_outputdata = model->duplicateData(
                        Subtensors_outputdata, "@Sub_OutputData",
                        Real_sub_outputdataDesc);

                DimValues Sub_outputdatasOffsets;
                Sub_outputdatasOffsets.set(Dim::N, dilationXInd * Sub_output_dilationX_dimenion / dilationX);
                Sub_outputdatasOffsets.set(Dim::W,
                        (dilationYInd * Sub_output_dilationY_dimenion / dilationY) * Real_sub_outputdataDesc.dim(Dim::W));

                V_Sub_outputdata.emplace_back(Real_sub_outputdata);
                V_Sub_outputdatasOffsets.emplace_back(Sub_outputdatasOffsets);
            }
        }

        auto SplitPermutedInputDataStage = _stageBuilder->addSplitStage(model,
                stage->name() + "@Split-Permuted-InputData", stage->origNode(),
                std::move(V_Sub_inputdatasOffsets), Reshape_Permuted_inputdata,
                V_Sub_inputdata);

        // sub tensors convolution
        for (int Sub_output_XInd = 0; Sub_output_XInd < Sub_output_dilationX_dimenion; ++Sub_output_XInd) {
            for (int Sub_output_YInd = 0; Sub_output_YInd < Sub_output_dilationY_dimenion;
                    ++Sub_output_YInd) {
            // Add SubDataConv stage
            auto Sub_outputdataDesc = Real_sub_outputdataDesc;
            Sub_outputdataDesc.setDim(Dim::W, Sub_outputdata_width);

            Data Sub_outputdata;
            Sub_outputdata = model->duplicateData(Subtensors_outputdata,
                    "@Sub_OutputData", Sub_outputdataDesc);

            auto newStage = model->addNewStage<StubStage>(
                    stage->origLayerName() + "@SubDataConv",
                    StageType::StubConv, stage->origNode(), {
                            V_Sub_inputdata[Sub_output_XInd * Sub_output_dilationY_dimenion + Sub_output_YInd],
                            weights,
                            biases,
                            scales }, { Sub_outputdata });

            newStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            newStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            newStage->attrs().set<int>("kernelStrideX", kernelStrideX_new);
            newStage->attrs().set<int>("kernelStrideY", kernelStrideY_new);

            newStage->attrs().set<int>("padLeft", padLeft_new);
            newStage->attrs().set<int>("padRight", padRight_new);

            newStage->attrs().set<int>("padTop", padTop_new);
            newStage->attrs().set<int>("padBottom", padBottom_new);
            newStage->attrs().set<int>("dilationX", 1);
            newStage->attrs().set<int>("dilationY", 1);
            newStage->attrs().set<int>("groupSize", groupSize);
            newStage->attrs().set<bool>("tryHW", true);

            newStage->attrs().set<float>("scaleFactor", scaleFactor);

            if (Sub_outputdata_expand) {
                _stageBuilder->addCropStage(model,
                        stage->name() + "@SubConvOutputData",
                        stage->origNode(), Sub_outputdata,
                        V_Sub_outputdata[Sub_output_XInd * Sub_output_dilationY_dimenion + Sub_output_YInd]);
            } else {
                V_Sub_outputdata[Sub_output_XInd * Sub_output_dilationY_dimenion + Sub_output_YInd] = Sub_outputdata;
            }
        }
        }

        auto Reshape_Permuted_outputdata_Desc = Subtensors_outputdata->desc();

        Reshape_Permuted_outputdata_Desc.setDim(Dim::H,
                Subtensors_outputdata->desc().dim(Dim::H) / Sub_output_dilationY_dimenion);
        Reshape_Permuted_outputdata_Desc.setDim(Dim::W,
                Subtensors_outputdata->desc().dim(Dim::W) * Sub_output_dilationY_dimenion);

        auto Reshape_Permuted_outputdata = model->duplicateData(
                Subtensors_outputdata, "@Reshape-Permuted-outputdata",
                Reshape_Permuted_outputdata_Desc);

        auto n = 0;
        auto c = 0;
        auto h = 0;
        auto w = 0;
        V_Sub_outputdatasOffsets[0].get(Dim::N, n);
        V_Sub_outputdatasOffsets[0].get(Dim::C, c);
        V_Sub_outputdatasOffsets[0].get(Dim::H, h);
        V_Sub_outputdatasOffsets[0].get(Dim::W, w);

        auto ConcatSubOutputDataStage = _stageBuilder->addConcatStage(model,
                stage->name() + "@Concat-Sub-OutputData", stage->origNode(),
                std::move(V_Sub_outputdatasOffsets), V_Sub_outputdata,
                Reshape_Permuted_outputdata);

        _stageBuilder->addReshapeStage(model,
                stage->name() + "@Reshape-conv-outputdata", stage->origNode(),
                Reshape_Permuted_outputdata, Subtensors_outputdata);

        // output permute
        DataDesc permute_outputdataDesc(DataType::FP16, DimsOrder::NCHW,
                { Subtensors_outputdata->desc().dim(Dim::N),
                        Subtensors_outputdata->desc().dim(Dim::W),
                        Subtensors_outputdata->desc().dim(Dim::H),
                        Subtensors_outputdata->desc().dim(Dim::C) });

        auto permute_outputdata = model->duplicateData(Subtensors_outputdata,
                "@Permuted-OutputData", permute_outputdataDesc);

        _stageBuilder->addPermuteStage(
            model,
            stage->origLayerName() + "@Permute-OutputData",
            stage->origNode(),
            Subtensors_outputdata,
            permute_outputdata,
            DimValues_<Dim>{{Dim::W, Dim::N}, {Dim::H, Dim::W}, {Dim::C, Dim::H}, {Dim::N, Dim::C}});

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
                    stage->origNode(), permute_outputdata,
                    Reinterpret_outputdata);

            auto CropToOutputDataStage = _stageBuilder->addCropStage(model,
                    stage->name() + "@crop-to-OutputData", stage->origNode(),
                    Reinterpret_outputdata, output);

        } else {
            _stageBuilder->addReshapeStage(model,
                    stage->name() + "@copy-Permute-OutputData",
                    stage->origNode(), permute_outputdata, output);
        }
        model->removeStage(stage);
    }
}
}  // namespace

Pass::Ptr PassManager::reshapeDilationConv() {
    return std::make_shared < PassImpl > (_stageBuilder);
}

}  // namespace vpu
