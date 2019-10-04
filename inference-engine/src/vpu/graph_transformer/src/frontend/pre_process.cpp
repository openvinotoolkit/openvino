// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vector>
#include <memory>
#include <string>

#include <details/caseless.hpp>
#include <cpp/ie_cnn_network.h>
#include <precision_utils.h>
#include <ie_parallel.hpp>

#include <vpu/sw/utility.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class MeanImageContent final : public CalculatedDataContent {
public:
    explicit MeanImageContent(const ie::PreProcessInfo& info) : _info(info) {}

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const override {
        auto countElem = _desc.dim(Dim::W) * _desc.dim(Dim::H) * _desc.dim(Dim::C);

        if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
            countElem *= 2;
        }

        return countElem * sizeof(fp16_t);
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>&, void* tempBuf) const override {
        VPU_PROFILE(MeanImageContent);

        auto numOfChannel = _info.getNumberOfChannels();

        auto imagePixels = _desc.dim(Dim::W) * _desc.dim(Dim::H);
        auto countElem = _desc.dim(Dim::W) * _desc.dim(Dim::H) * _desc.dim(Dim::C);

        auto dstPtr = static_cast<fp16_t*>(tempBuf);
        auto dstPtr2 = dstPtr;

        if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
            dstPtr2 += countElem;
        }

        ie::parallel_for(numOfChannel, [=](int i) {
            auto meanDataBlob = _info[i]->meanData;

            ie::PrecisionUtils::f32tof16Arrays(
                dstPtr2 + i * imagePixels,
                meanDataBlob->buffer().as<const float*>(),
                imagePixels,
                -1.0f);
        });

        if (_desc.dimsOrder() == DimsOrder::NHWC || _desc.dimsOrder() == DimsOrder::HWC) {
            kchw_to_hwck(dstPtr2, dstPtr, _desc);
        }
    }

private:
    ie::PreProcessInfo _info;
};

class MeanValueContent final : public CalculatedDataContent {
public:
    explicit MeanValueContent(const ie::PreProcessInfo& info) : _info(info) {}

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const override {
        return _info.getNumberOfChannels() * sizeof(fp16_t);
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>&, void* tempBuf) const override {
        VPU_PROFILE(MeanValueContent);

        IE_ASSERT(_desc.totalDimSize() == _info.getNumberOfChannels());

        auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(_info.getNumberOfChannels(), [dstPtr, this](int i) {
            dstPtr[i] = ie::PrecisionUtils::f32tof16(-_info[i]->meanValue);
        });
    }

private:
    ie::PreProcessInfo _info;
};

}  // namespace

void FrontEnd::addPreProcessStages(const Model::Ptr& model) {
    VPU_PROFILE(addPreProcessStages);

    const auto& env = CompileEnv::get();

    for (const auto& inputInfo : _ieNetworkParser.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& preProcess = netInput->getPreProcess();

        if (preProcess.getMeanVariant() != ie::NONE) {
            auto input = getVpuData(ieData);
            IE_ASSERT(input != nullptr);

            int numOfChannel = preProcess.getNumberOfChannels();

            env.log->debug("add pre-processing for input %s", input->name());

            if (preProcess.getMeanVariant() == ie::MEAN_IMAGE) {
                auto meanImage = model->addConstData(
                    input->name() + "@mean-image",
                    input->desc(),
                    std::make_shared<MeanImageContent>(preProcess));

                auto newInput = model->duplicateData(
                    input,
                    "@after-mean-image");

                bindData(newInput, ieData);

                _stageBuilder->addSumStage(
                    model,
                    meanImage->name(),
                    nullptr,
                    input, meanImage,
                    newInput);

                input = newInput;
            } else {
                auto meanValues = model->addConstData(
                    input->name() + "@mean-values",
                    DataDesc({numOfChannel}),
                    std::make_shared<MeanValueContent>(preProcess));

                auto newInput = model->duplicateData(
                    input,
                    "@after-mean-values");

                bindData(newInput, ieData);

                _stageBuilder->addBiasStage(
                    model,
                    meanValues->name(),
                    nullptr,
                    input, meanValues,
                    newInput);

                input = newInput;
            }

            if (preProcess[0]->stdScale != 1.0f) {
                for (int i = 1; i < numOfChannel; i++) {
                    if (!isFloatEqual(preProcess[i - 1]->stdScale, preProcess[i]->stdScale)) {
                        VPU_THROW_EXCEPTION << "Different values of stdScale are not supported";
                    }
                }

                auto newInput = model->duplicateData(
                    input,
                    "@after-std-scale");

                bindData(newInput, ieData);

                _stageBuilder->addPowerStage(
                    model,
                    input->name() + "@stdScale=" + std::to_string(preProcess[0]->stdScale),
                    nullptr,
                    preProcess[0]->stdScale,
                    1.0f,
                    0.0f,
                    input,
                    newInput);

                input = newInput;
            }
        }
    }
}

}  // namespace vpu
