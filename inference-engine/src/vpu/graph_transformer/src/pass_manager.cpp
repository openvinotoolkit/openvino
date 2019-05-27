// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/pass_manager.hpp>

#include <sstream>
#include <iomanip>
#include <memory>
#include <string>

#include <vpu/compile_env.hpp>

namespace vpu {

//
// PerStagePass
//

void PerStagePass::run(const Model::Ptr& model) {
    for (const auto& stage : model->getStages()) {
        if (_types.count(stage->type()) == 0) {
            continue;
        }

        runForStage(model, stage);
    }
}

//
// PassSet
//

void PassSet::run(const Model::Ptr& model) const {
    using MilliSecondsFP64 = std::chrono::duration<double, std::milli>;

    const auto& env = CompileEnv::get();

    env.log->debug("Run passes");
    VPU_LOGGER_SECTION(env.log);

    for (const auto& pass : _passes) {
        auto pass_ind = &pass - &_passes.front();
        auto pass_start_time = std::chrono::high_resolution_clock::now();

        model->cleanUpDatas();
        pass->run(model);

        auto pass_end_time = std::chrono::high_resolution_clock::now();

        env.log->debug(
            "[PASS %m%d / %d] duration : %f ms",
            std::setw(2), pass_ind + 1, _passes.size(),
            std::chrono::duration_cast<MilliSecondsFP64>(pass_end_time - pass_start_time).count());
    }

    model->cleanUpDatas();
}

//
// PassManager
//

PassSet::Ptr PassManager::buildMiddleEnd() {
    const auto& env = CompileEnv::get();

    auto passes = std::make_shared<PassSet>();

    //
    // Initial state
    //

    _dumpInd = 0;
    passes->addPass(dumpModel("initial"));

    //
    // To overcome fp16 limitations
    //

    if (env.config.hwOptimization) {
        if (env.config.hwAdaptiveMode) {
            passes->addPass(analyzeWeightableLayers());
        } else {
            if (!env.netConfig.hasManualDataScale()) {
                passes->addPass(estimateSingleNetworkScale());
            }

            passes->addPass(propagateDataScale());
        }

        passes->addPass(dumpModel("dataScaling"));
    }

    passes->addPass(findSubGraphs());
    passes->addPass(dumpModel("findSubGraphs"));

    //
    // Model common adaptation
    //

    passes->addPass(splitGroupedConv());
    passes->addPass(dumpModel("splitGroupedConv"));

    //
    // Model HW-specific optimizations
    //

    if (env.config.hwOptimization) {
        passes->addPass(replaceFCbyConv());
        passes->addPass(dumpModel("replaceFCbyConv"));

        passes->addPass(replaceDeconvByConv());
        passes->addPass(dumpModel("replaceDeconvByConv"));

        passes->addPass(swapConcatAndHwOps());
        passes->addPass(dumpModel("swapConcatAndHwOps"));

        passes->addPass(mergeHwStages());
        passes->addPass(dumpModel("mergeHwStages"));

        passes->addPass(splitHwDepthConv());
        passes->addPass(dumpModel("splitHwDepthConv"));

        passes->addPass(splitHwConvAndPool());
        passes->addPass(dumpModel("splitHwConvAndPool"));
    }

    passes->addPass(hwPadding());
    passes->addPass(dumpModel("hwPadding"));

    //
    // Batch support
    //

    passes->addPass(adjustDataBatch());
    passes->addPass(dumpModel("adjustDataBatch"));

    //
    // HW stages tiling
    //

    if (env.config.hwOptimization) {
        passes->addPass(hwConvTiling());
        passes->addPass(hwPoolTiling());
        passes->addPass(hwFullyConnectedTiling());
        passes->addPass(dumpModel("hwTiling"));
    }

    //
    // Model SW-specific adaptation
    //

    passes->addPass(swConvAdaptation());
    passes->addPass(swDeconvAdaptation());
    passes->addPass(swPoolAdaptation());
    passes->addPass(swFullyConnectedAdaptation());
    passes->addPass(dumpModel("swAdaptation"));

    //
    // Model SW-specific optimizations
    //

    passes->addPass(mergeReLUAndBias());
    passes->addPass(dumpModel("mergeReLUAndBias"));

    //
    // Data layout adjustment
    //

    passes->addPass(adjustDataLayout());
    passes->addPass(dumpModel("adjustDataLayout"));

    //
    // Model special stages processing
    //

    passes->addPass(processSpecialStages());
    passes->addPass(dumpModel("processSpecialStages"));

    //
    // Data location adjustment
    //

    passes->addPass(adjustDataLocation());
    passes->addPass(dumpModel("adjustDataLocation"));

    //
    // Model common optimizations
    //

    if (env.config.copyOptimization.getOrDefault(true)) {
        passes->addPass(eliminateCopyStages());
        passes->addPass(dumpModel("eliminateCopyStages"));
    }

    //
    // HW/SW injection
    //

    if (env.config.hwOptimization && env.config.injectSwOps.getOrDefault(true)) {
        passes->addPass(injectSw());
        passes->addPass(dumpModel("injectSw"));
    }

    //
    // Final resource allocation
    //

    passes->addPass(allocateResources());
    passes->addPass(dumpModel("allocateResources"));

    //
    // HW stages finalization
    //

    if (env.config.hwOptimization) {
        passes->addPass(finalizeHwOps());
        passes->addPass(dumpModel("hwFinalization"));
    }

    //
    // Final check
    //

    passes->addPass(finalCheck());

    return passes;
}

//
// DumpPass
//

namespace {

class DumpPass final : public Pass {
public:
    DumpPass(const std::string& postfix,
             const BackEnd::Ptr& backEnd) :
            _postfix(postfix), _backEnd(backEnd) {
    }

    void run(const Model::Ptr& model) override {
        _backEnd->dumpModel(model, _postfix);
    }

private:
    std::string _postfix;
    BackEnd::Ptr _backEnd;
};

}  // namespace

Pass::Ptr PassManager::dumpModel(const std::string& postfix) {
    std::ostringstream ostr;
    ostr << std::setw(2) << std::setfill('0') << _dumpInd << "-" << postfix;

    ++_dumpInd;

    return std::make_shared<DumpPass>(ostr.str(), _backEnd);
}

}  // namespace vpu
