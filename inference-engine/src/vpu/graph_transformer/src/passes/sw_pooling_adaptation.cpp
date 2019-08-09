// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <vector>
#include <unordered_set>
#include <memory>
#include <set>

namespace vpu {

namespace {

class PoolStage final : public StageNode {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<PoolStage>(*this);
    }

    void propagateScaleFactorsImpl(
            const SmallVector<float>&,
            ScalePropagationStep) override {
        VPU_THROW_EXCEPTION << "Must never be called";
    }

    void propagateDataOrderImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto finalOrder = input->desc().dimsOrder();
        if (input->desc().dim(Dim::N, 1) > 1) {
            // To merge batch into channels
            finalOrder = finalOrder.createMovedDim(Dim::C, 2);
        }

        _orderInfo.setInput(_inputEdges[0], finalOrder);
        _orderInfo.setOutput(_outputEdges[0], finalOrder);
    }

    void getDataStridesRequirementsImpl() const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();

        auto dimsOrder = input->desc().dimsOrder();

        StridesRequirement reqs;

        if (input->desc().dim(Dim::N, 1) > 1) {
            // To merge batch into previous dimension.
            reqs.add(dimsOrder.dimInd(Dim::N), DimStride::Compact);
        }

        _stridesInfo.setInput(_inputEdges[0], reqs);
        _stridesInfo.setOutput(_outputEdges[0], reqs);

        //
        // * AvgPool/MaxPool support both YXZ and ZYX orders:
        //   * ZYX versions support both input and output strides.
        //   * YXZ versions support only output strides.
        // * GlobalPooling supports both 3D/4D layouts.
        //

        if (_type == StageType::MaxPool || _type == StageType::AvgPool) {
            if (dimsOrder.dimInd(Dim::C) == 0) {
                _stridesInfo.setInput(_inputEdges[0], StridesRequirement::compact());
            }
        }
    }

    void finalizeDataLayoutImpl() override {
    }

    void getBatchSupportInfoImpl() const  override {
        // Pooling will support batch by merging it with previous dimension.
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
        auto excludePad = attrs().get<bool>("excludePad");

        serializer.append(static_cast<uint32_t>(kernelSizeX));
        serializer.append(static_cast<uint32_t>(kernelSizeY));
        serializer.append(static_cast<uint32_t>(kernelStrideX));
        serializer.append(static_cast<uint32_t>(kernelStrideY));
        serializer.append(static_cast<uint32_t>(padLeft));
        serializer.append(static_cast<uint32_t>(padTop));
        serializer.append(static_cast<uint32_t>(excludePad));
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        IE_ASSERT(_inputEdges.size() == 1);
        IE_ASSERT(_outputEdges.size() == 1);

        auto input = _inputEdges[0]->input();
        auto output = _outputEdges[0]->output();

        if (_type == StageType::GlobalMaxPool ||
            _type == StageType::GlobalAvgPool) {
            input->serializeNewBuffer(serializer);
            output->serializeNewBuffer(serializer);
        } else {
            auto perm = input->desc().dimsOrder().toPermutation();
            IE_ASSERT(perm.size() == 4);
            IE_ASSERT(perm.back() == Dim::N);

            perm.pop_back();

            input->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::fromPermutation(perm),
                {
                    {perm[2], {perm[2], Dim::N}},
                    {perm[1], {perm[1]}},
                    {perm[0], {perm[0]}}
                });

            output->serializeOldBuffer(
                handle_from_this(),
                serializer,
                DimsOrder::fromPermutation(perm),
                {
                    {perm[2], {perm[2], Dim::N}},
                    {perm[1], {perm[1]}},
                    {perm[0], {perm[0]}}
                });
        }
    }
};

class PassImpl final : public Pass {
public:
    void run(const Model::Ptr& model) override;
};

void PassImpl::run(const Model::Ptr& model) {
    VPU_PROFILE(swPoolAdaptation);

    for (const auto& stage : model->getStages()) {
        if (stage->type() != StageType::StubMaxPool &&
            stage->type() != StageType::StubAvgPool) {
            continue;
        }

        auto input = stage->input(0);
        auto output = stage->output(0);

        auto kernelSizeX = stage->attrs().get<int>("kernelSizeX");
        auto kernelSizeY = stage->attrs().get<int>("kernelSizeY");
        auto kernelStrideX = stage->attrs().get<int>("kernelStrideX");
        auto kernelStrideY = stage->attrs().get<int>("kernelStrideY");
        auto padLeft = stage->attrs().get<int>("padLeft");
        auto padRight = stage->attrs().get<int>("padRight");
        auto padTop = stage->attrs().get<int>("padTop");
        auto padBottom = stage->attrs().get<int>("padBottom");
        auto excludePad = stage->attrs().get<bool>("excludePad");

        model->disconnectStageDatas(stage);

        auto stageType = StageType::None;
        if (stage->type() == StageType::StubMaxPool) {
            if (padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0 &&
                output->desc().dim(Dim::W) == 1 && output->desc().dim(Dim::H) == 1) {
                stageType = StageType::GlobalMaxPool;
            } else {
                stageType = StageType::MaxPool;
            }
        } else {
            if (padLeft == 0 && padRight == 0 && padTop == 0 && padBottom == 0 &&
                output->desc().dim(Dim::W) == 1 && output->desc().dim(Dim::H) == 1) {
                stageType = StageType::GlobalAvgPool;
            } else {
                stageType = StageType::AvgPool;
            }
        }

        auto swStage = model->addNewStage<PoolStage>(
            stage->name(),
            stageType,
            stage->origLayer(),
            {input},
            {output});

        swStage->attrs().set<int>("kernelSizeX", kernelSizeX);
        swStage->attrs().set<int>("kernelSizeY", kernelSizeY);

        swStage->attrs().set<int>("kernelStrideX", kernelStrideX);
        swStage->attrs().set<int>("kernelStrideY", kernelStrideY);

        swStage->attrs().set<int>("padLeft", padLeft);
        swStage->attrs().set<int>("padRight", padRight);
        swStage->attrs().set<int>("padTop", padTop);
        swStage->attrs().set<int>("padBottom", padBottom);

        swStage->attrs().set<bool>("excludePad", excludePad);

        model->removeStage(stage);
    }
}

}  // namespace

Pass::Ptr PassManager::swPoolAdaptation() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
