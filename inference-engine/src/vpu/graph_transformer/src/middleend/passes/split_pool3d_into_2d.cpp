// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"
#include "vpu/stage_builder.hpp"
#include "vpu/utils/numeric.hpp"
#include "precision_utils.h"
#include "vpu/model/data_contents/ie_blob_content.hpp"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vpu {

namespace {

enum PoolNDMethod   { PoolND_max = 1, PoolND_avg = 2 };

enum PoolNDRounding { PoolND_floor = 3, PoolND_ceil  = 4 };

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

// Split 3D pooling into the subgraph of 2D pooling operations
// Ultimate goal is to enable/optimize the I3D networks on VPU
//
// Generally, not every 3D pooling allows naive such reducing,
// but fortunately, the pooling layers found in the I3D network
// for IE belong to the special cases that allow reducing.
//
// Here is the list of the pooling layers found at the I3D:
//
//     Kernel   Method Strides  Pads   Can reduce?
//     -----------------------------------------------
//     1, 3, 3   max   1, 2, 2  same   yes: 2D in fact
//     3, 3, 3   max   1, 1, 1  same   yes: reduce max
//     3, 3, 3   max   2, 2, 2  same   yes: reduce max
//     2, 2, 2   max   2, 2, 2  same   yes: reduce max
//     2, 7, 7   avg   1, 1, 1  valid  yes: no pads
//       9, 1    avg     1, 1               2D already
//     -----------------------------------------------
//
// So, we only need to reduce the following quite simple cases:
// - If pooling method is "max", so any pads do not matter
// - If method is "avg", but there is no pads in fact
// - If pooling involves only H, W axes (2D in fact)
void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitPool3DInto2D);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::PoolND) {
            continue;
        }

        using PV = typename ie::PropertyVector<unsigned int>;

        const auto& kernel_shape = stage->attrs().get<PV>("kernel_shape");
        const auto& pads_begin   = stage->attrs().get<PV>("pads_begin");
        const auto& pads_end     = stage->attrs().get<PV>("pads_end");
        const auto& strides      = stage->attrs().get<PV>("strides");

        // NB: parameters `interleaved` and `rounding_type` are not used (yet)
        const auto pooling_method = stage->attrs().get<int>("pooling_method");
        const auto exclude_pad    = stage->attrs().get<int>("exclude_pad");

        const auto try_hw = stage->attrs().get<int>("try_hw");

        auto kernelNDims = kernel_shape.size();
        if (kernelNDims != 3) {
            continue;
        }

        if (pooling_method == PoolND_avg && exclude_pad) {
            bool zeroPads = pads_begin[2] == 0 &&
                            pads_end[2] == 0;
            if (!zeroPads && kernel_shape[2] != 0) {
                continue;  // cannot correctly split into 2D poolings
            }
        }

        VPU_THROW_UNLESS(stage->numInputs() == 1, "unsupported number of inputs: %d", stage->numInputs());
        VPU_THROW_UNLESS(stage->numOutputs() == 1, "unsupported number of outputs: %d", stage->numOutputs());

        auto input = stage->input(0);
        auto output = stage->output(0);

        DataDesc inputDesc = input->desc();
        DataDesc outputDesc = output->desc();

        if (inputDesc.type() != DataType::FP16) {
            continue;
        }

        if (inputDesc.dimsOrder() != DimsOrder::NCDHW) {
            continue;
        }

        VPU_THROW_UNLESS(kernelNDims == pads_begin.size(),
                         "incompatible: kernel ndims=%d, pads ndims=%lu", kernelNDims, pads_begin.size());
        VPU_THROW_UNLESS(kernelNDims == pads_end.size(),
                         "incompatible: kernel ndims=%d, pads ndims=%lu", kernelNDims, pads_end.size());
        VPU_THROW_UNLESS(kernelNDims == strides.size(),
                         "incompatible: kernel ndims=%d, strides ndims=%lu", kernelNDims, strides.size());

        VPU_THROW_UNLESS(inputDesc.type() == outputDesc.type(), "incompatible data types");
        VPU_THROW_UNLESS(inputDesc.dimsOrder() == outputDesc.dimsOrder(), "incompatible dim orders");

        int I_N = inputDesc.dim(Dim::N);
        int IC = inputDesc.dim(Dim::C);
        int ID = inputDesc.dim(Dim::D);
        int IH = inputDesc.dim(Dim::H);
        int IW = inputDesc.dim(Dim::W);

        int ON = outputDesc.dim(Dim::N);
        int OC = outputDesc.dim(Dim::C);
        int OD = outputDesc.dim(Dim::D);
        int OH = outputDesc.dim(Dim::H);
        int OW = outputDesc.dim(Dim::W);

        VPU_THROW_UNLESS(I_N == ON, "incompatible: input batch=%d, output batch=%d", I_N, ON);
        VPU_THROW_UNLESS(IC == OC, "incompatible: input channels=%d, output channels=%d", IC, OC);

        // check spacial dims of output
        int  inputShape[] = {IW, IH, ID, IC, I_N};
        int outputShape[] = {OW, OH, OD, OC, ON};
        for (int i = 0; i < 3; i++) {
            int expectedOutputSize = (inputShape[i]
                                    + pads_begin[i] + pads_end[i]
                                    - kernel_shape[i]) / strides[i] + 1;
            VPU_THROW_UNLESS(outputShape[i] == expectedOutputSize,
                             "falied check of output shape: i=%d, actual=%d, expected=%d",
                             i, outputShape[i], expectedOutputSize);
        }

        int KW = kernel_shape[0];
        int KH = kernel_shape[1];
        int KD = kernel_shape[2];

        // Replace the PoolND stage with sub-graph of 2D pooling stub stages
        //
        // Create the sub-graph in reverse order: from output to input node
        // Create only those sub-tensors really required for the sub-graph

        model->disconnectStage(stage);

        // output = scale * preScaled,
        // if pooling method is "avg" and output tensor depth is non-trivial
        //
        // preScaled not required, if method is "max" of output depth is 1
        bool needScale = pooling_method == PoolND_avg && KD > 1;
        Data preScaled = nullptr;
        Data scales = nullptr;
        if (needScale) {
            // Prepare the `scales` tensor:
            // Despite we need the scalar scaling factor,
            // use the Scale layer with one scale per output channel.
            // In our case, we setup all these scale factors equally.

            const auto generator = [KD, OC](const ie::Blob::Ptr& blob) {
                auto scalesPtr = blob->buffer().as<fp16_t*>();
                auto scaleFactor = ie::PrecisionUtils::f32tof16(1.f / static_cast<float>(KD));
                std::fill(scalesPtr, scalesPtr + OC, scaleFactor);
            };

            scales = model->addConstData(stage->name() + "@scales", DataDesc({OC}), generator);

            // Add the scaling stage
            preScaled = model->duplicateData(output, "@pre_scaled", outputDesc);
            _stageBuilder->addScaleStage(model,
                                         stage->name() + "@scale",
                                         stage->origLayer(),
                                         preScaled,
                                         scales,
                                         output);
        }

        // preScaled = merge{subOutputs[d], d=0,...,D-1} -- was split by `depth` axis
        // or:
        // output = merge{subOutputs[d], d=0,...,D-1} -- if final scaling is not needed
        Data preFinal = needScale ? preScaled : output;
        DataVector subOutputs3D(OD);
        for (int d = 0; d < OD; d++) {
            auto postfix = formatString("@output_depth=%d/%d@3D", d + 1, OD);
            DataDesc subOutputsDesc(outputDesc.type(), DimsOrder::NCDHW, {OW, OH, 1, OC, ON});
            subOutputs3D[d] = model->duplicateData(preFinal, postfix, subOutputsDesc);
        }
        _stageBuilder->addConcatStage(model,
                                      stage->name() + "@concat",
                                      stage->origLayer(),
                                      Dim::D,
                                      subOutputs3D,
                                      preFinal);

        // subOutputs3D[d] = reshape(subOutputs[d]) -- add `depth` axis
        DataVector subOutputs(OD);
        for (int d = 0; d < OD; d++) {
            auto postfix = formatString("@output_depth=%d/%d", d + 1, OD);
            DataDesc subOutputsDesc(outputDesc.type(), DimsOrder::NCHW, {OW, OH, OC, ON});
            subOutputs[d] = model->duplicateData(preFinal, postfix, subOutputsDesc);
            _stageBuilder->addReshapeStage(model,
                                           stage->name() + "@reshape",
                                           stage->origLayer(),
                                           subOutputs[d],
                                           subOutputs3D[d]);
        }

        // subOutputs[d] = sum(subPool[d+k], k=0, ..., K-1)
        // or:
        // subOutputs[d] = max(subPool[d+k], k=0, ..., K-1)
        //
        // Up to ID intermediate results of 2D pooling:
        // a subPool[d] tensor per each d=0, ..., ID-1
        //
        // Up to OD x KD temporary results for summation
        // (or taking "max")
        std::vector<Data> subPool(ID);
        std::vector<std::vector<Data>> subTemp(OD);
        for (int d = 0; d < OD; d++) {
            subTemp[d].resize(KD);

            // Given the output depth `d`, check all k=0, ..., K-1
            //
            // Check if the corresponding input depth `i` fits into
            // the boubdaries of the input tensor
            //
            // For every such (d, k), create the 2D tensor for the
            // result of the partial pooling (if no such yet)
            //
            // Also create the stage for summation (or taking max)
            int iCount = 0;
            int iPrev = -1;
            int iLast = -1;
            int kLast = -1;
            for (int k = 0; k < KD; k++) {
                int i = d * strides[2] + k - pads_begin[2];
                if (i < 0 || i >= ID) {
                    continue;  // index is out of input's bounds
                }

                iCount++;  // process this `k`
                iPrev = iLast;
                iLast = i;
                kLast = k;

                // create subPool and subTemp data items, if no such yet
                DataDesc subDesc(outputDesc.type(), DimsOrder::NCHW, {OW, OH, OC, ON});
                if (subPool[i] == nullptr) {
                    auto postfix = formatString("@sub_pool=%d/%d", i + 1, ID);
                    subPool[i] = model->duplicateData(subOutputs[d], postfix, subDesc);
                }
                {
                    auto postfix = formatString("@temp(d=%d/%d,k=%d/%d)", d + 1, OD, k + 1, KD);
                    subTemp[d][k] = model->duplicateData(subPool[i], postfix, subDesc);
                }

                if (iCount < 2) {
                    continue;  // need at least two active i's for summation of subPool[i]
                }

                if (pooling_method == PoolND_avg) {
                    auto postfix = formatString("@sum(d=%d/%d,k=%d/%d)", d + 1, OD, k + 1, KD);
                    _stageBuilder->addSumStage(model,
                                               stage->name() + postfix,
                                               stage->origLayer(),
                                               iCount == 2 ?
                                                   subPool[iPrev] :  // if 1st + 2nd sum,
                                                   subTemp[d][k-1],  // if 3rd or further
                                               subPool[iLast],
                                               subTemp[d][k]);
                } else if (pooling_method == PoolND_max) {
                    auto postfix = formatString("@max(d=%d/%d,k=%d/%d)", d + 1, OD, k + 1, KD);
                    _stageBuilder->addMaxStage(model,
                                               stage->name() + postfix,
                                               stage->origLayer(),
                                               iCount == 2 ?
                                                   subPool[iPrev] :  // if 1st + 2nd sum,
                                                   subTemp[d][k-1],  // if 3rd or further
                                               subPool[iLast],
                                               subTemp[d][k]);
                } else {
                    VPU_THROW_UNLESS(pooling_method == PoolND_avg ||
                                     pooling_method == PoolND_max,
                                     "unsupported pooling method: %d", pooling_method);
                }
            }
            VPU_THROW_UNLESS(iCount > 0, "software bug (please report): iCount=%d", iCount);

            _stageBuilder->addCopyStage(model,
                                        stage->name() + "@copy",
                                        stage->origLayer(),
                                        iCount == 1 ?
                                            subPool[iLast] :    // if single subPool
                                            subTemp[d][kLast],  // if two or more...
                                        subOutputs[d],
                                        "splitPool3DInto2D");
        }

        // subInputs[d] = input(:,:,d,:,:) -- split by `depth` axis
        DataVector subInputs(ID);

        // subPool[i] = Pool2D(subInputs[i])
        for (int i = 0; i < ID; i++) {
            if (subPool[i] == nullptr) {
                continue;  // this subPool[i] is not not needed
            }

            // create subInputs[i], if it was not created previously
            if (subInputs[i] == nullptr) {
                auto postfix = formatString("@input_depth=%d/%d", i + 1, ID);
                DataDesc subInputsDesc(inputDesc.type(), DimsOrder::NCHW, {IW, IH, IC, I_N});
                subInputs[i] = model->duplicateData(input, postfix, subInputsDesc);
            }

            using PT = ie::PoolingLayer::PoolType;
            PT poolType = pooling_method == PoolND_avg ? PT::AVG : PT::MAX;

            auto postfix = formatString("@pool2d(d=%d/%d)", i + 1, ID);
            Stage pool2d = _stageBuilder->addPoolingStage(model,
                                                          stage->name() + postfix,
                                                          stage->origLayer(),
                                                          subInputs[i],
                                                          subPool[i],
                                                          poolType);

            pool2d->attrs().set<int>("kernelSizeX", KW);
            pool2d->attrs().set<int>("kernelSizeY", KH);

            pool2d->attrs().set<int>("kernelStrideX", strides[0]);
            pool2d->attrs().set<int>("kernelStrideY", strides[1]);

            pool2d->attrs().set<int>("padLeft", pads_begin[0]);
            pool2d->attrs().set<int>("padRight", pads_end[0]);
            pool2d->attrs().set<int>("padTop", pads_begin[1]);
            pool2d->attrs().set<int>("padBottom", pads_end[1]);

            pool2d->attrs().set<bool>("excludePad", exclude_pad != 0);

            pool2d->attrs().set<bool>("tryHW", try_hw != 0);
        }

        // subInputs[d] = squeeze(subInputs3D[d])
        //
        // Note that some of subInputs[d] may be not actually needed
        DataVector subInputs3D(ID);
        for (int d = 0; d < ID; d++) {
            if (subInputs[d] == nullptr) {
                continue;  // this subInputs[d] is not needed
            }
            auto postfix = formatString("@input_depth=%d/%d", d + 1, ID);
            DataDesc subInputsDesc3D(inputDesc.type(), DimsOrder::NCDHW, {IW, IH, 1, IC, I_N});
            subInputs3D[d] = model->duplicateData(input, postfix + "@3D", subInputsDesc3D);
            _stageBuilder->addReshapeStage(model,
                                           stage->name() + "@split",
                                           stage->origLayer(),
                                           subInputs3D[d],
                                           subInputs[d]);
        }

        // Same as addSplitStage() but allow null pointers in outputs.
        // Such null pointers would mean that some outputs not needed,
        // so we exclude them from final list for the Split stage.
        // Assume all outputs have same length along the given axis.
        auto myAddSplitStage = [](vpu::StageBuilder::Ptr _stageBuilder,
                                  const vpu::Model       & model,
                                  const std::string      & name,
                                  const  ie::CNNLayerPtr & layer,
                                  const vpu::Dim           axis,
                                  const vpu::Data        & input,
                                  const vpu::DataVector  & outputs) {
            std::vector<DimValues> offsets;
            std::vector<Data> actualOutputs;

            int outputsNum = outputs.size();
            int inputAxisLen = input->desc().dim(axis);
            int outputAxisLen = inputAxisLen / outputsNum;

            DimValues curOffset({{axis, 0}});
            for (const auto& output : outputs) {
                if (output != nullptr) {
                    VPU_THROW_UNLESS(output->desc().dim(axis) == outputAxisLen,
                                     "incompatible output dim: actual=%d, expected=%d",
                                     output->desc().dim(axis), outputAxisLen);
                    actualOutputs.push_back(output);
                    offsets.push_back(curOffset);
                }
                curOffset.set(axis, curOffset[axis] + outputAxisLen);
            }

            VPU_THROW_UNLESS(!actualOutputs.empty(), "no actual outputs");
            auto stage = _stageBuilder->addSplitStage(model,
                                                      name,
                                                      layer,
                                                      std::move(offsets),
                                                      input,
                                                      actualOutputs);
            stage->attrs().set("axis", axis);

            return stage;
        };

        myAddSplitStage(_stageBuilder,
                        model,
                        stage->name() + "@split",
                        stage->origLayer(),
                        Dim::D,
                        input,
                        subInputs3D);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::splitPool3DInto2D() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
