// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/middleend/sw/utility.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/model/data_contents/mean_contents.hpp>

#include <cpp/ie_cnn_network.h>
#include <precision_utils.h>

#include <vector>
#include <memory>
#include <string>

namespace vpu {

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
                std::make_shared<MeanImageContent>(preProcess, input->desc()));

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
                input->name() + "@stdScale=" + InferenceEngine::CNNLayer::ie_serialize_float(preProcess[0]->stdScale),
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
