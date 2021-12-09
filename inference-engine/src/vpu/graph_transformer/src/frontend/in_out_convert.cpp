// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::addDataTypeConvertStages(const Model& model) {
    VPU_PROFILE(addDataTypeConvertStages);

    const auto& env = CompileEnv::get();

    env.log->trace("Add Data type conversion stages");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& input : model->datas()) {
        if (input->usage() != DataUsage::Input) {
            continue;
        }

        env.log->trace("Input : %s [%s]", input, input->desc().type());
        VPU_LOGGER_SECTION(env.log);

        switch (input->desc().type()) {
            case DataType::U8:
            case DataType::FP32: {
                env.log->trace("Convert to FP16");

                auto fp16Desc = input->desc();
                fp16Desc.setType(DataType::FP16);

                const auto inputFP16 = model->duplicateData(
                        input,
                        "@FP16",
                        fp16Desc);

                input->attrs().set<Data>("fp16_copy", inputFP16);

                bindData(inputFP16, input->origData());

                for (const auto consumerEdge : input->consumerEdges()) {
                    model->replaceStageInput(consumerEdge, inputFP16);
                }

                _stageBuilder->createConvertStage(
                        model,
                        inputFP16->name(),
                        input,
                        inputFP16,
                        1.0f,
                        0.0f);

                break;
            }

            default: {
                // Nothing to do.
                break;
            }
        }
    }

    for (const auto& output : model->datas()) {
        if (output->usage() != DataUsage::Output) {
            continue;
        }

        env.log->trace("Output : %s [%s]", output, output->desc().type());
        VPU_LOGGER_SECTION(env.log);

        if (output->desc().type() != DataType::FP32) {
            // Output datas keep their precision (intermeadiate have been forced to FP16 in case of FP32 from IR).
            // If FP32 output has been requested VPU executes in FP16 with following convert FP16 -> FP32
            continue;
        }

        env.log->trace("Convert from FP16");

        auto fp16Desc = output->desc();
        fp16Desc.setType(DataType::FP16);

        const auto outputFP16 = model->duplicateData(
            output,
            "@FP16",
            fp16Desc);

        output->attrs().set<Data>("fp16_copy", outputFP16);

        bindData(outputFP16, output->origData());

        if (const auto producerEdge = output->producerEdge()) {
            model->replaceStageOutput(producerEdge, outputFP16);
        }

        const auto stage = _stageBuilder->createConvertStage(
            model,
            outputFP16->name(),
            outputFP16,
            output);

        const auto withDetectionOutput = model->attrs().getOrDefault<bool>("withDetectionOutput", false);
        stage->attrs().set<bool>("convertFromDetOutput", withDetectionOutput);

        const auto haveBatch = model->batchSize() != 1 && _unbatchedOutputs.count(output->origData()) == 0;
        stage->attrs().set<bool>("haveBatch", haveBatch);
    }
}

}  // namespace vpu
