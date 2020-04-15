// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/middleend/pass_manager.hpp"
#include "vpu/utils/numeric.hpp"
#include "vpu/model/data_contents/ie_blob_content.hpp"

#include "precision_utils.h"
#include "ie_memcpy.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vpu {

namespace {

//----------------------------------------------------------------------

// Fulfill the 4D subWeights tensor of OIYX layout
// by copying the kd'th hyperplane of the original
// 5D weights tensor having the OIZYX layout.
// Here X is width, Y is height, Z is depth axis.
static void copySubWeights(fp16_t subWeightsPtr[],
                     const fp16_t weightsPtr[],
                     const int    weightsShape[],
                     const int    kd) {
    int KW = weightsShape[0];  // width
    int KH = weightsShape[1];  // height
    int KD = weightsShape[2];  // depth
    int KI = weightsShape[3];  // input channels
    int KO = weightsShape[4];  // output channels (times groups if any)
    VPU_THROW_UNLESS(0 <= kd && kd < KD, "index out of bound: kd=%d, KD=%d", kd, KD);

    int nCubes     = KI * KO;
    int  cubeSize  = KW * KH * KD;
    int planeSize  = KW * KH;
    auto planeBytes = static_cast<size_t >(KW * KH * sizeof(fp16_t));

    for (int q = 0; q < nCubes; q++) {
              fp16_t *dstPlane = subWeightsPtr + q * planeSize;
        const fp16_t *srcPlane =    weightsPtr + q *  cubeSize
                                              + kd * planeSize;
        ie_memcpy(dstPlane, planeBytes, srcPlane, planeBytes);
    }
}

//----------------------------------------------------------------------

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(splitConv3DInto2D);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::ConvND) {
            continue;
        }

        using PV = typename ie::PropertyVector<unsigned int>;

        auto pads_begin = stage->attrs().get<PV>("pads_begin");
        auto pads_end   = stage->attrs().get<PV>("pads_end");
        auto strides    = stage->attrs().get<PV>("strides");
        auto dilations  = stage->attrs().get<PV>("dilations");
        auto groups     = stage->attrs().get<int>("groups");
        auto try_hw     = stage->attrs().get<int>("try_hw");

        int kernelNDims = pads_begin.size();
        VPU_THROW_UNLESS(kernelNDims == pads_end.size(),
                         "wrong pads ndims=%lu, expected=%d", pads_end.size(), kernelNDims);
        VPU_THROW_UNLESS(kernelNDims == strides.size(),
                         "wrong strides ndims=%lu, expected=%d", strides.size(), kernelNDims);
        VPU_THROW_UNLESS(kernelNDims == dilations.size(),
                         "wrong dilations ndims=%lu, expected=%d", dilations.size(), kernelNDims);
        VPU_THROW_UNLESS(groups >= 1, "wrong number of groups=%d", groups);

        if (kernelNDims != 3) {
            continue;
        }

        VPU_THROW_UNLESS(stage->numInputs() == 3, "wrong number of inputs: %d", stage->numInputs());
        VPU_THROW_UNLESS(stage->numOutputs() == 1, "wrong number of outputs: %d", stage->numOutputs());

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto output = stage->output(0);

        DataDesc inputDesc = input->desc();
        DataDesc outputDesc = output->desc();

        VPU_THROW_UNLESS(inputDesc.type() == outputDesc.type(),
                         "input and output types must equal, but: input type=%d, output type=%d",
                         inputDesc.type(), outputDesc.type());
        if (inputDesc.type() != DataType::FP16) {
            continue;
        }

        VPU_THROW_UNLESS(inputDesc.dimsOrder() == outputDesc.dimsOrder(),
                         "input and output dim orders must equal");
        if (inputDesc.dimsOrder() != DimsOrder::NCDHW) {
            continue;
        }

        int IN = inputDesc.dim(Dim::N);
        int IC = inputDesc.dim(Dim::C);
        int ID = inputDesc.dim(Dim::D);
        int IH = inputDesc.dim(Dim::H);
        int IW = inputDesc.dim(Dim::W);

        int ON = outputDesc.dim(Dim::N);
        int OC = outputDesc.dim(Dim::C);
        int OD = outputDesc.dim(Dim::D);
        int OH = outputDesc.dim(Dim::H);
        int OW = outputDesc.dim(Dim::W);

        DataDesc weightsDesc = weights->desc();
        VPU_THROW_UNLESS(weightsDesc.type() == DataType::FP16, "wrong weights type: %d", weightsDesc.type());
        VPU_THROW_UNLESS(weightsDesc.numDims() == 5, "wrong number of weights dims: %d", weightsDesc.numDims());

        int KW = weightsDesc.dim(Dim::W);
        int KH = weightsDesc.dim(Dim::H);
        int KD = weightsDesc.dim(Dim::D);
        int KI = weightsDesc.dim(Dim::C);  // TODO: define Dim::I and Dim::O,
        int KO = weightsDesc.dim(Dim::N);  //       or rework as dim-agnostic

        VPU_THROW_UNLESS(KI == IC / groups,
                         "kernel 'inputs' dim must equal to number of input channels per group, "
                         "but: KI=%d, IC=%d, groups=%d", KI, IC, groups);
        VPU_THROW_UNLESS(KO == OC, "kernel 'output' dim must equal to number of output channels, "
                         "but: KO=%d, OC=%d", KO, OC);

        // check spacial dims of output
        int   inputShape[] = {IW, IH, ID, IC, IN};
        int  outputShape[] = {OW, OH, OD, OC, ON};
        int weightsShape[] = {KW, KH, KD, KI, KO};
        for (int i = 0; i < 3; i++) {
            int dilatedKernelSize = dilations[i] * (weightsShape[i] - 1) + 1;
            int expectedOutputSize = (inputShape[i]
                                    + pads_begin[i] + pads_end[i]
                                    - dilatedKernelSize) / strides[i] + 1;
            VPU_THROW_UNLESS(outputShape[i] == expectedOutputSize,
                             "failed check of output shape: i=%d, actual=%d, expected=%d",
                             i, outputShape[i], expectedOutputSize);
        }

        //
        // Replace the ConvND stage with sub-graph of 2D ConvStub stages
        //
        // Create the sub-graph in reverse order: from output to input node
        // Create only those sub-tensors really required for the sub-graph
        //

        model->disconnectStage(stage);

        // output = bias + preBiased
        Data preBiased = model->duplicateData(output, "@pre_biased", outputDesc);
        _stageBuilder->addBiasStage(model,
                                    stage->name() + "@bias",
                                    stage->origLayer(),
                                    preBiased,
                                    biases,
                                    output);

        // preBiased = merge{subOutputs[d], d=0,...,D-1} -- was split by `depth` axis
        DataVector subOutputs3D(OD);
        for (int d = 0; d < OD; d++) {
            auto postfix = formatString("@output_depth=%d/%d@3D", d + 1, OD);
            DataDesc subOutputsDesc(outputDesc.type(), DimsOrder::NCDHW, {OW, OH, 1, OC, ON});
            subOutputs3D[d] = model->duplicateData(preBiased, postfix, subOutputsDesc);
        }
        _stageBuilder->addConcatStage(model,
                                      stage->name() + "@concat",
                                      stage->origLayer(),
                                      Dim::D,
                                      subOutputs3D,
                                      preBiased);

        // subOutputs3D[d] = reshape(subOutputs[d]) -- add `depth` axis
        DataVector subOutputs(OD);
        for (int d = 0; d < OD; d++) {
            auto postfix = formatString("@output_depth=%d/%d", d + 1, OD);
            DataDesc subOutputsDesc(outputDesc.type(), DimsOrder::NCHW, {OW, OH, OC, ON});
            subOutputs[d] = model->duplicateData(preBiased, postfix, subOutputsDesc);
            _stageBuilder->addReshapeStage(model,
                                           stage->name() + "@reshape",
                                           stage->origLayer(),
                                           subOutputs[d],
                                           subOutputs3D[d]);
        }

        // subOutputs[d] = sum(subConv[d,k], k=0,...,K-1)
        //
        // Up to OD x KD intermediate resulting tensors:
        // note that some of them may be null, if input
        // index for (d, k) misses tensor's boundaries
        //
        // We also need auxiliary tensors for summation:
        // as the Sum stage only supports two inputs
        std::vector<std::vector<vpu::Data>> subConv(OD);
        std::vector<std::vector<vpu::Data>> subSum(OD);
        for (int d = 0; d < OD; d++) {
            subConv[d].resize(KD);  // initialized with null pointers
            subSum[d].resize(KD);

            // Given the output depth `d`, check all k=0,...,K-1
            // Check if the corresponding input depth `i` fits into
            // the boubdaries of the input tensor
            //
            // For every such (d, k), create the 2D tensor for the
            // result of the partial convolution
            //
            // Also create the stage for summation of these 2D conv
            // results by 'k'
            int kCount = 0;
            int kLast = -1;
            int kPrev = -1;
            for (int k = 0; k < KD; k++) {
                int i = d * strides[2]    // strided output index
                      + k * dilations[2]  // dilated kernel index
                      - pads_begin[2];    // pads begin for depth

                if (i < 0 || i >= ID) {
                    continue;  // index is out of input's bounds
                }

                // create 2D convolution data item, if no such yet
                if (subConv[d][k] == nullptr) {
                    VPU_THROW_UNLESS(subSum[d][k] == nullptr, "software bug: d=%d, k=%k", d, k);
                    auto postfix = formatString("@kernel_depth=%d/%d", k + 1, KD);
                    DataDesc subDesc(outputDesc.type(), DimsOrder::NCHW, {OW, OH, OC, ON});
                    subConv[d][k] = model->duplicateData(subOutputs[d], postfix, subDesc);
                    subSum[d][k] = model->duplicateData(subOutputs[d], postfix + "@sum", subDesc);
                }

                kCount++;  // process this `k`
                kPrev = kLast;
                kLast = k;

                if (kCount < 2) {
                    continue;  // need at least two active k's for summation of  subConv[d][k]
                }

                auto postfix = formatString("@sum(d=%d/%d,k=%d/%d)", d + 1, OD, k + 1, KD);
                _stageBuilder->addSumStage(model,
                                           stage->name() + postfix,
                                           stage->origLayer(),
                                           kCount == 2 ?
                                               subConv[d][kPrev] :  // if 1st + 2nd summ
                                               subSum[d][kPrev],    // if 3rd or further
                                           subConv[d][kLast],
                                           subSum[d][kLast]);
            }
            VPU_THROW_UNLESS(kCount >= 1, "software bug: kCount=%d", kCount);

            _stageBuilder->addCopyStage(model,
                                        stage->name() + "@copy",
                                        stage->origLayer(),
                                        kCount == 1 ?
                                            subConv[d][kLast] :  // if single subConv
                                            subSum[d][kLast],    // if two or more...
                                        subOutputs[d],
                                        "splitConv3DInto2D");
        }

        // subInputs[d] = input(:,:,d,:,:) -- split by `depth` axis
        DataVector subInputs(ID);

        // subWeights[k] = weights(:,:,k,:,:) -- split by `depth` axis of kernel
        DataVector subWeights(KD);

        // subConv[d,k] = Conv2D(subWeights[d], subInputs[d+k])
        //
        // Here: 4D subWeights[d] is d'th plane of 5D weights,
        // and subInputs[d] is 4D plane of the 5D input tensor
        //
        // Note that some such 2D convolutions may be undefined
        // if i'th plane is out of the bounds of the input tensor
        // (where i = d + k in simplest case)
        //
        // Also note that some of the 2D convolutions may occur
        // not needed for computing of the output if nontrivial
        // strides and/or dilations
        for (int d = 0; d < OD; d++) {
            for (int k = 0; k < KD; k++) {
                int i = d * strides[2]    // strided output index
                        + k * dilations[2]  // dilated kernel index
                        - pads_begin[2];    // pads begin for depth

                if (i < 0 || i >= ID) {
                    continue;  // index is out of input's bounds
                }

                if (subConv[d][k] == nullptr) {
                    continue;  // this (d, k) is not not needed
                }

                // create subInputs[i], if it was not created previously
                if (subInputs[i] == nullptr) {
                    auto postfix = formatString("@input_depth=%d/%d", i + 1, ID);
                    DataDesc subInputsDesc(inputDesc.type(), DimsOrder::NCHW, {IW, IH, IC, IN});
                    subInputs[i] = model->duplicateData(input, postfix, subInputsDesc);
                }

                // create subWeights[k], if not created previously
                if (subWeights[k] == nullptr) {
                    auto postfix = formatString("@kernel_depth=%d/%d", d + 1, KD);

                    auto weightsContent = weights->content();
                    VPU_THROW_UNLESS(weightsContent != nullptr, "need weights content");

                    auto weightsPtr = weightsContent->get<fp16_t>();
                    VPU_THROW_UNLESS(weightsPtr != nullptr, "cannot get weights data");

                    ie::SizeVector subWeightsDims = {static_cast<size_t>(KW), static_cast<size_t>(KH),
                                                     static_cast<size_t>(KI), static_cast<size_t>(KO)};
                    ie::TensorDesc subWeightsDesc(ie::Precision::FP16, subWeightsDims, ie::Layout::NCHW);
                    auto subWeightsBlob = ie::make_shared_blob<fp16_t>(subWeightsDesc);
                    subWeightsBlob->allocate();

                    auto subWeightsPtr = subWeightsBlob->buffer().as<fp16_t*>();
                    copySubWeights(subWeightsPtr, weightsPtr, weightsShape, k);

                    subWeights[k] = model->duplicateData(weights, postfix,
                                                         DataDesc({KW, KH, KI, KO}),
                                                         ieBlobContent(subWeightsBlob));
                }

                auto postfix = formatString("@conv2d(d=%d/%d,k=%d/%d)", d + 1, OD, k + 1, KD);
                Stage conv2d = _stageBuilder->addConvolutionStage(model,
                                                                  stage->name() + postfix,
                                                                  stage->origLayer(),
                                                                  subInputs[i],
                                                                  subConv[d][k],
                                                                  subWeights[k],
                                                                  model->addFakeData(),   // biases
                                                                  model->addFakeData());  // scales

                conv2d->attrs().set<int>("kernelSizeX", KW);
                conv2d->attrs().set<int>("kernelSizeY", KH);

                conv2d->attrs().set<int>("kernelStrideX", strides[0]);
                conv2d->attrs().set<int>("kernelStrideY", strides[1]);

                conv2d->attrs().set<int>("padLeft", pads_begin[0]);
                conv2d->attrs().set<int>("padRight", pads_end[0]);
                conv2d->attrs().set<int>("padTop", pads_begin[1]);
                conv2d->attrs().set<int>("padBottom", pads_end[1]);

                conv2d->attrs().set<int>("dilationX", dilations[0]);
                conv2d->attrs().set<int>("dilationY", dilations[1]);

                conv2d->attrs().set<int>("groupSize", groups);

                conv2d->attrs().set<bool>("tryHW", try_hw != 0);
            }
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
            DataDesc subInputsDesc3D(inputDesc.type(), DimsOrder::NCDHW, {IW, IH, 1, IC, IN});
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
                                     "wrong output shape");
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

Pass::Ptr PassManager::splitConv3DInto2D() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
