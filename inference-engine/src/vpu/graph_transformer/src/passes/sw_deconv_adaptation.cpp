// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include <set>

#include <ie_parallel.hpp>

#include <vpu/sw/utility.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace {

void depthDeconvolutionRelayoutCHW(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int channels) {
    ie::parallel_for3d(channels, KY, KX, [=](int c, int ky, int kx) {
        int iidx = c * KX * KY + ky * KX + kx;
        IE_ASSERT(iidx >= 0 && iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = c * KX * KY + inv_ky * KX + inv_kx;
        IE_ASSERT(oidx >= 0 && oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

class DepthDeconvolutionCHWWeightsContent final : public CalculatedDataContent {
public:
    DepthDeconvolutionCHWWeightsContent(
            const DataContent::Ptr& origContent,
            int KX, int KY, int channels) :
            CalculatedDataContent({origContent}),
            _KX(KX), _KY(KY), _channels(channels) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(DepthDeconvolutionCHWWeightsContent);
        depthDeconvolutionRelayoutCHW(
            baseContents[0]->get<fp16_t>(), _desc.totalDimSize(),
            static_cast<fp16_t*>(tempBuf), _desc.totalDimSize(),
            _KX, _KY, _channels);
    }

private:
    int _KX;
    int _KY;
    int _channels;
};

void depthDeconvolutionRelayoutHWC(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int channels) {
    ie::parallel_for3d(channels, KY, KX, [=](int c, int ky, int kx) {
        int iidx = c * KX * KY + ky * KX + kx;
        IE_ASSERT(iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = inv_ky * KX * channels + inv_kx * channels + c;
        IE_ASSERT(oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

class DepthDeconvolutionHWCWeightsContent final : public CalculatedDataContent {
public:
    DepthDeconvolutionHWCWeightsContent(
            const DataContent::Ptr& origContent,
            int KX, int KY, int channels) :
            CalculatedDataContent({origContent}),
            _KX(KX), _KY(KY), _channels(channels) {
    }

protected:
    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(DepthDeconvolutionHWCWeightsContent);
        depthDeconvolutionRelayoutHWC(
            baseContents[0]->get<fp16_t>(), _desc.totalDimSize(),
            static_cast<fp16_t*>(tempBuf), _desc.totalDimSize(),
            _KX, _KY, _channels);
    }

private:
    int _KX;
    int _KY;
    int _channels;
};

void deconvolutionRelayout(
    const fp16_t* src, int src_size,
    fp16_t* dst, int dst_size,
    int KX, int KY,
    int IC, int OC) {
    ie::parallel_for4d(OC, IC, KY, KX, [=](int oc, int ic, int ky, int kx) {
        int iidx = ic * OC * KY * KX
                 + oc * KY * KX
                 + ky * KX
                 + kx;
        IE_ASSERT(iidx >= 0 && iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = oc * IC * KY * KX
                 + ic * KY * KX
                 + inv_ky * KX
                 + inv_kx;
        IE_ASSERT(oidx >=  0 && oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

class DeconvolutionWeightsContent final : public CalculatedDataContent {
public:
    DeconvolutionWeightsContent(
            const DataContent::Ptr& origContent,
            int KX, int KY,
            int IC, int OC) :
            CalculatedDataContent({origContent}),
            _KX(KX), _KY(KY),
            _IC(IC), _OC(OC) {
    }

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const override {
        return 2 * _desc.totalDimSize() * sizeof(fp16_t);
    }


    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>& baseContents, void* tempBuf) const override {
        VPU_PROFILE(DeconvolutionWeightsContent);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);
        auto dstPtr2 = dstPtr + _desc.totalDimSize();

        deconvolutionRelayout(
            baseContents[0]->get<fp16_t>(), _desc.totalDimSize(),
            dstPtr2, _desc.totalDimSize(),
            _KX, _KY,
            _IC, _OC);

        kchw_to_hwkc(dstPtr2, dstPtr, _desc);
    }

private:
    int _KX;
    int _KY;
    int _IC;
    int _OC;
};

class DeconvStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<DeconvStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto finalOrder = input->desc().dimsOrder();
        if (finalOrder.dimInd(Dim::C) == 1) {
            // HCW -> CHW
            finalOrder.moveDim(Dim::C, 2);
        }

        if (_type == StageType::DepthDeconv) {
            if (finalOrder != input->desc().dimsOrder()) {
                _orderInfo.setInput(_inputEdges[0], finalOrder);
            }
            _orderInfo.setOutput(_outputEdges[0], finalOrder);
        } else {
            _orderInfo.setInput(_inputEdges[0], finalOrder.createMovedDim(Dim::C, 0));
            _orderInfo.setOutput(_outputEdges[0], finalOrder.createMovedDim(Dim::C, 0));
        }
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto finalOrder = input->desc().dimsOrder();
        if (finalOrder.dimInd(Dim::C) == 1) {
            // HCW -> CHW
            finalOrder.moveDim(Dim::C, 2);
        }

        if (_type == StageType::DepthDeconv) {
            if (finalOrder.dimInd(Dim::C) == 0) {
                // HWC
                _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
                _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
            }
        } else {
            _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
            _stridesInfo.setOutput(_outputEdges[0], StridesRequirement::compact());
        }
    }

    void finalizeDataLayoutImpl() override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto output = _outputEdges[0]->output();

        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");

        Data swWeights;

        if (_type == StageType::DepthDeconv) {
            if (input->desc().dimsOrder().dimInd(Dim::C) == 0) {
                //
                // HWC case
                //

                swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
                if (swWeights == nullptr) {
                    DataDesc newWeightsDesc({
                        kernelSizeX * kernelSizeY,
                        1,
                        output->desc().dim(Dim::C)});

                    swWeights = _model->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<DepthDeconvolutionHWCWeightsContent>(
                            weights->content(),
                            kernelSizeX, kernelSizeY,
                            output->desc().dim(Dim::C)));

                    weights->attrs().set<Data>("swWeights", swWeights);
                }
            } else if (input->desc().dimsOrder().dimInd(Dim::C) == 2) {
                //
                // CHW case
                //

                swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
                if (swWeights == nullptr) {
                    DataDesc newWeightsDesc({
                        kernelSizeX * kernelSizeY,
                        1,
                        output->desc().dim(Dim::C)});

                    swWeights = _model->duplicateData(
                        weights,
                        "@SW",
                        newWeightsDesc,
                        std::make_shared<DepthDeconvolutionCHWWeightsContent>(
                            weights->content(),
                            kernelSizeX, kernelSizeY,
                            output->desc().dim(Dim::C)));

                    weights->attrs().set<Data>("swWeights", swWeights);
                }
            }
        } else {
            swWeights = weights->attrs().getOrDefault<Data>("swWeights", nullptr);
            if (swWeights == nullptr) {
                DataDesc newWeightsDesc({
                    kernelSizeX * kernelSizeY,
                    input->desc().dim(Dim::C),
                    output->desc().dim(Dim::C)});

                swWeights = _model->duplicateData(
                    weights,
                    "@SW",
                    newWeightsDesc,
                    std::make_shared<DeconvolutionWeightsContent>(
                        weights->content(),
                        kernelSizeX, kernelSizeY,
                        input->desc().dim(Dim::C),
                        output->desc().dim(Dim::C)));

                weights->attrs().set<Data>("swWeights", swWeights);
            }
        }

        IE_ASSERT(swWeights != nullptr);

        _model->replaceStageInput(_inputEdges[1], swWeights);
    }

    void getBatchSupportInfoImpl() const  override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        _batchInfo.setInput(_inputEdges[0], BatchSupport::Split);
        _batchInfo.setOutput(_outputEdges[0], BatchSupport::Split);
    }

    void finalCheckImpl() const override {
    }

    void serializeParamsImpl(BlobSerializer& serializer) const override {
        auto kernelSizeX = attrs().get<int>("kernelSizeX");
        auto kernelSizeY = attrs().get<int>("kernelSizeY");
        auto kernelStrideX = attrs().get<int>("kernelStrideX");
        auto kernelStrideY = attrs().get<int>("kernelStrideY");
        auto padLeft = attrs().get<int>("padLeft");
        auto padTop = attrs().get<int>("padTop");
        auto dilationX = attrs().get<int>("dilationX");
        auto dilationY = attrs().get<int>("dilationY");

        serializer.append(static_cast<uint32_t>(kernelSizeX));
        serializer.append(static_cast<uint32_t>(kernelSizeY));
        serializer.append(static_cast<uint32_t>(kernelStrideX));
        serializer.append(static_cast<uint32_t>(kernelStrideY));
        serializer.append(static_cast<uint32_t>(padLeft));
        serializer.append(static_cast<uint32_t>(padTop));
        serializer.append(static_cast<uint32_t>(dilationX));
        serializer.append(static_cast<uint32_t>(dilationY));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 3);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto weights = _inputEdges[1]->input();
        auto biases = _inputEdges[2]->input();
        auto output = _outputEdges[0]->output();

        input->serializeOldBuffer(handle_from_this(), serializer);
        output->serializeOldBuffer(handle_from_this(), serializer);
        weights->serializeOldBuffer(handle_from_this(), serializer);

        if (!_tempBufferEdges.empty()) {
            _tempBufferEdges[0]->tempBuffer()->serializeOldBuffer(handle_from_this(), serializer);
        }

        // TODO: remove this
        biases->serializeOldBuffer(handle_from_this(), serializer);
    }
};

class PassImpl final : public Pass {
public:
    explicit PassImpl(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    void run(const Model::Ptr& model) override;

private:
    StageBuilder::Ptr _stageBuilder;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(swDeconvAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubDeconv)
            continue;

        auto input = stage->input(0);
        auto weights = stage->input(1);
        auto biases = stage->input(2);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto dilationX = stage->attrs().get<int>("dilationX");
        auto dilationY = stage->attrs().get<int>("dilationY");
        auto groupSize = stage->attrs().get<int>("groupSize");

        model->disconnectStageDatas(stage);

        if (groupSize == 0 ||
            (groupSize > input->desc().dim(Dim::C)) ||
            (input->desc().dim(Dim::C) % groupSize != 0) ||
            (groupSize > output->desc().dim(Dim::C)) ||
            (output->desc().dim(Dim::C) % groupSize != 0)) {
            VPU_THROW_EXCEPTION << "DeconvolutionLayer has invalid group value";
        }

        if (groupSize == 1) {
            if (biases->usage() != DataUsage::Fake) {
                auto tempOutput = model->duplicateData(
                    output,
                    "@temp");

                _stageBuilder->addBiasStage(
                    model,
                    stage->name() + "@biases",
                    stage->origLayer(),
                    tempOutput, biases,
                    output);

                output = tempOutput;
            }

            auto swStage = model->addNewStage<DeconvStage>(
                stage->name(),
                StageType::Deconvolution,
                stage->origLayer(),
                {input, weights, biases},
                {output});

            swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

            swStage->attrs().set<int>("padLeft", padLeft);
            swStage->attrs().set<int>("padRight", padRight);
            swStage->attrs().set<int>("padTop", padTop);
            swStage->attrs().set<int>("padBottom", padBottom);

            swStage->attrs().set<int>("dilationX", dilationX);
            swStage->attrs().set<int>("dilationY", dilationY);
        } else if (groupSize == input->desc().dim(Dim::C) &&
                   groupSize == output->desc().dim(Dim::C)) {
            if (biases->usage() != DataUsage::Fake) {
                auto tempOutput = model->duplicateData(
                    output,
                    "@temp");

                _stageBuilder->addBiasStage(
                    model,
                    stage->name() + "@biases",
                    stage->origLayer(),
                    tempOutput, biases,
                    output);

                output = tempOutput;
            }

            auto swStage = model->addNewStage<DeconvStage>(
                stage->name(),
                StageType::DepthDeconv,
                stage->origLayer(),
                {input, weights, biases},
                {output});

            swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
            swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

            swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
            swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

            swStage->attrs().set<int>("padLeft", padLeft);
            swStage->attrs().set<int>("padRight", padRight);
            swStage->attrs().set<int>("padTop", padTop);
            swStage->attrs().set<int>("padBottom", padBottom);

            swStage->attrs().set<int>("dilationX", dilationX);
            swStage->attrs().set<int>("dilationY", dilationY);
        } else {
            VPU_THROW_EXCEPTION << "Internal error : grouped deconvolution was not processed";
        }

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swDeconvAdaptation() {
    return std::make_shared<PassImpl>(_stageBuilder);
}

}  // namespace vpu
