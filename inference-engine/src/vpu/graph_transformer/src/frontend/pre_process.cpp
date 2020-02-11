// Copyright (C) 2018-2020 Intel Corporation
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

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

class MeanImageContent final : public CalculatedDataContent {
public:
    explicit MeanImageContent(const ie::PreProcessInfo& info) : _info(info) {
    }

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const override {
        size_t countElem = checked_cast<size_t>(desc().dim(Dim::W) * desc().dim(Dim::H) * desc().dim(Dim::C));
        if (desc().dimsOrder() == DimsOrder::NHWC || desc().dimsOrder() == DimsOrder::HWC) {
            countElem *= 2;
        }

        return countElem * sizeof(fp16_t);
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>&, void* tempBuf) const override {
        VPU_PROFILE(MeanImageContent);

        const size_t numOfChannel = _info.getNumberOfChannels();

        const size_t imagePixels = checked_cast<size_t>(desc().dim(Dim::W) * desc().dim(Dim::H));
        const size_t countElem = checked_cast<size_t>(desc().dim(Dim::W) * desc().dim(Dim::H) * desc().dim(Dim::C));

        const auto dstPtr = static_cast<fp16_t*>(tempBuf);

        auto dstPtr2 = dstPtr;
        if (desc().dimsOrder() == DimsOrder::NHWC || desc().dimsOrder() == DimsOrder::HWC) {
            dstPtr2 += countElem;
        }

        ie::parallel_for(numOfChannel, [=](size_t i) {
            const auto meanDataBlob = _info[i]->meanData;

            ie::PrecisionUtils::f32tof16Arrays(
                dstPtr2 + i * imagePixels,
                meanDataBlob->buffer().as<const float*>(),
                imagePixels,
                -1.0f);
        });

        if (desc().dimsOrder() == DimsOrder::NHWC || desc().dimsOrder() == DimsOrder::HWC) {
            kchw_to_hwck(dstPtr2, dstPtr, desc());
        }
    }

private:
    ie::PreProcessInfo _info;
};

class MeanValueContent final : public CalculatedDataContent {
public:
    explicit MeanValueContent(const ie::PreProcessInfo& info) : _info(info) {
    }

protected:
    size_t getTempBufSize(const SmallVector<DataContent::Ptr, 2>&) const override {
        return _info.getNumberOfChannels() * sizeof(fp16_t);
    }

    void fillTempBuf(const SmallVector<DataContent::Ptr, 2>&, void* tempBuf) const override {
        VPU_PROFILE(MeanValueContent);

        IE_ASSERT(checked_cast<size_t>(desc().totalDimSize()) == _info.getNumberOfChannels());

        const auto dstPtr = static_cast<fp16_t*>(tempBuf);

        ie::parallel_for(_info.getNumberOfChannels(), [dstPtr, this](size_t i) {
            dstPtr[i] = ie::PrecisionUtils::f32tof16(-_info[i]->meanValue);
        });
    }

private:
    ie::PreProcessInfo _info;
};

}  // namespace

void FrontEnd::addPreProcessStages(const Model& model) {
    VPU_PROFILE(addPreProcessStages);

    const auto& env = CompileEnv::get();

    env.log->trace("Process network pre-processing section");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& inputInfo : _ieParsedNetwork.networkInputs) {
        const auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        const auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        const auto& preProcess = netInput->getPreProcess();
        if (preProcess.getMeanVariant() == ie::NONE) {
            continue;
        }

        auto input = getVpuData(ieData);
        IE_ASSERT(input != nullptr);

        env.log->trace("Add pre-processing for input %s", input->name());
        VPU_LOGGER_SECTION(env.log);

        if (preProcess.getMeanVariant() == ie::MEAN_IMAGE) {
            env.log->trace("MEAN_IMAGE");
            VPU_LOGGER_SECTION(env.log);

            const auto meanImage = model->addConstData(
                input->name() + "@mean-image",
                input->desc(),
                std::make_shared<MeanImageContent>(preProcess));

            const auto newInput = model->duplicateData(
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
            env.log->trace("MEAN_VALUE");
            VPU_LOGGER_SECTION(env.log);

            const int numOfChannel = checked_cast<int>(preProcess.getNumberOfChannels());

            const auto meanValues = model->addConstData(
                input->name() + "@mean-values",
                DataDesc({numOfChannel}),
                std::make_shared<MeanValueContent>(preProcess));

            const auto newInput = model->duplicateData(
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
            env.log->trace("STD_SCALE");
            VPU_LOGGER_SECTION(env.log);

            const size_t numOfChannel = preProcess.getNumberOfChannels();

            for (size_t i = 1; i < numOfChannel; i++) {
                if (!isFloatEqual(preProcess[i - 1]->stdScale, preProcess[i]->stdScale)) {
                    VPU_THROW_FORMAT("Different per-channel values of stdScale array are not supported");
                }
            }

            const auto newInput = model->duplicateData(
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

}  // namespace vpu
