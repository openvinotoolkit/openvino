// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <set>
#include <memory>
#include <unordered_set>

#include <vpu/compile_env.hpp>
#include <vpu/middleend/hw/utility.hpp>
#include <vpu/middleend/sw/utility.hpp>
#include <vpu/configuration/options/hw_pool_conv_merge.hpp>

namespace vpu {

namespace {

Stage getNextPoolStage(const Stage& stage, const Data& output) {
    auto input = stage->input(0);

    if (input->desc().dim(Dim::W) % 2 != 0 ||
        input->desc().dim(Dim::H) % 2 != 0 ||
        output->desc().dim(Dim::W) % 2 != 0 ||
        output->desc().dim(Dim::H) % 2 != 0) {
        return nullptr;
    }

    auto nextPool = getOneOfSingleNextStage(stage, {StageType::StubMaxPool});
    if (nextPool == nullptr) {
        return nullptr;
    }

    if (!nextPool->attrs().getOrDefault<bool>("tryHW", false)) {
        return nullptr;
    }

    auto poolOutput = nextPool->output(0);

    if (poolOutput->desc().dim(Dim::W) % 2 != 0 ||
        poolOutput->desc().dim(Dim::H) % 2 != 0) {
        return nullptr;
    }

    auto convKernelSizeX = stage->attrs().get<int>("kernelSizeX");
    auto convKernelSizeY = stage->attrs().get<int>("kernelSizeY");
    auto convKernelStride = stage->attrs().get<int>("kernelStrideX");
    auto convPadLeft = stage->attrs().get<int>("padLeft");
    auto convPadRight = stage->attrs().get<int>("padRight");
    auto convPadTop = stage->attrs().get<int>("padTop");
    auto convPadBottom = stage->attrs().get<int>("padBottom");

    auto poolKernelSizeX = nextPool->attrs().get<int>("kernelSizeX");
    auto poolKernelSizeY = nextPool->attrs().get<int>("kernelSizeY");
    auto poolKernelStride = nextPool->attrs().get<int>("kernelStrideX");
    auto poolPadLeft = nextPool->attrs().get<int>("padLeft");
    auto poolPadRight = nextPool->attrs().get<int>("padRight");
    auto poolPadTop = nextPool->attrs().get<int>("padTop");
    auto poolPadBottom = nextPool->attrs().get<int>("padBottom");

    // TODO: check which convolution and pooling parameters are supported

    if (convKernelSizeX == 3 && convKernelSizeY == 3 &&
        convKernelStride == 1 &&
        convPadLeft == 1 && convPadRight == 1 && convPadTop == 1 && convPadBottom == 1 &&
        poolKernelSizeX == 2 && poolKernelSizeY == 2 &&
        poolKernelStride == 2 &&
        poolPadLeft == 0 && poolPadRight == 0 && poolPadTop == 0 && poolPadBottom == 0) {
        return nextPool;
    }

    return nullptr;
}

class PassImpl final : public Pass {
public:
    void run(const Model& model) override;
};

void PassImpl::run(const Model& model) {
    VPU_PROFILE(mergeHwStages);

    const auto& env = CompileEnv::get();

    for (const auto& stage : model->getStages()) {
        if (stage == nullptr) {
            continue;
        }

        auto tryHW = stage->attrs().getOrDefault<bool>("tryHW", false);
        if (!tryHW) {
            continue;
        }

        IE_ASSERT(stage->numOutputs() == 1);

        auto output = stage->output(0);

        if (stage->type() == StageType::StubConv) {
            stage->attrs().set("origConvOutput", output->desc());
        }

        //
        // Try to merge next ReLU layer or Clamp
        //

        EnumSet<StageType> supportedPostOps{StageType::Relu};
        if (stage->type() == StageType::StubConv) {
            supportedPostOps.insert(StageType::LeakyRelu);
            supportedPostOps.insert(StageType::Clamp);
        }

        if (auto nextPostOpStage = getOneOfSingleNextStage(stage, supportedPostOps)) {
            bool isOK = true;

            if (nextPostOpStage->type() == StageType::Clamp) {
                auto min_value = nextPostOpStage->attrs().get<float>("min_value");

                if (!isFloatEqual(min_value, 0.0f)) {
                    isOK = false;
                }
            }

            if (nextPostOpStage->type() == StageType::LeakyRelu) {
                auto negativeSlope = nextPostOpStage->attrs().get<float>("negativeSlope");

                if (!isFloatEqual(negativeSlope, 0.0f)) {
                    // Only integer scales are supported

                    auto reverseScale = 1.0f / negativeSlope;

                    if (!isFloatEqual(std::fabs(std::ceil(reverseScale) - reverseScale), 0.0f)) {
                        isOK = false;
                    }
                }
            }

            if (isOK) {
                output = nextPostOpStage->output(0);

                model->disconnectStage(nextPostOpStage);

                model->replaceStageOutput(stage->outputEdge(0), output);

                if (nextPostOpStage->type() == StageType::Clamp) {
                    auto max_value = nextPostOpStage->attrs().get<float>("max_value");
                    stage->attrs().set<bool>("withClamp", true);
                    stage->attrs().set<float>("clampMax", max_value);
                } else {
                    auto negativeSlope = nextPostOpStage->attrs().get<float>("negativeSlope");

                    stage->attrs().set<bool>("withReLU", true);
                    stage->attrs().set<float>("negativeSlope", negativeSlope);
                    if (nextPostOpStage->type() == StageType::Relu) {
                        stage->attrs().set<uint32_t>("a0", 0);
                        stage->attrs().set<uint32_t>("a1", 1);
                        stage->attrs().set<float>("reluScale", 1.0f);
                    } else  {
                        stage->attrs().set<uint32_t>("a0", 1);
                        stage->attrs().set<uint32_t>("a1", static_cast<uint32_t>(1.0f / negativeSlope));
                        stage->attrs().set<float>("reluScale", negativeSlope);
                    }
                }

                model->removeStage(nextPostOpStage);
            }
        }

        //
        // Try to merge next Pooling layer
        //

        if (env.config.get<HwPoolConvMergeOption>()) {
            if (stage->type() == StageType::StubConv) {
                if (auto nextPoolStage = getNextPoolStage(stage, output)) {
                    output = nextPoolStage->output(0);

                    model->disconnectStage(nextPoolStage);

                    model->replaceStageOutput(stage->outputEdge(0), output);

                    auto poolKernelSizeX = nextPoolStage->attrs().get<int>("kernelSizeX");
                    auto poolKernelSizeY = nextPoolStage->attrs().get<int>("kernelSizeY");
                    auto poolKernelStride = nextPoolStage->attrs().get<int>("kernelStrideX");
                    auto poolPadLeft = nextPoolStage->attrs().get<int>("padLeft");
                    auto poolPadRight = nextPoolStage->attrs().get<int>("padRight");
                    auto poolPadTop = nextPoolStage->attrs().get<int>("padTop");
                    auto poolPadBottom = nextPoolStage->attrs().get<int>("padBottom");

                    stage->attrs().set<bool>("withPool", true);
                    stage->attrs().set<int>("poolKernelSizeX", poolKernelSizeX);
                    stage->attrs().set<int>("poolKernelSizeY", poolKernelSizeY);
                    stage->attrs().set<int>("poolKernelStride", poolKernelStride);
                    stage->attrs().set<int>("poolPadLeft", poolPadLeft);
                    stage->attrs().set<int>("poolPadRight", poolPadRight);
                    stage->attrs().set<int>("poolPadTop", poolPadTop);
                    stage->attrs().set<int>("poolPadBottom", poolPadBottom);

                    model->removeStage(nextPoolStage);
                }
            }
        }
    }
}

}  // namespace

Pass::Ptr PassManager::mergeHwStages() {
    return std::make_shared<PassImpl>();
}

}  // namespace vpu
