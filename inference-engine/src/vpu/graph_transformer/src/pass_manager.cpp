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

    int passInd = 0;
    for (const auto& p : _passes) {
        env.log->debug("Start pass %m%d / %d [%s]", std::setw(2), passInd + 1, _passes.size(), p.second);

        auto startTime = std::chrono::high_resolution_clock::now();

        model->cleanUp();

        p.first->run(model);

        auto endTime = std::chrono::high_resolution_clock::now();

        env.log->debug(
            "Pass %m%d / %d [%s] duration : %f ms",
            std::setw(2), passInd + 1, _passes.size(), p.second,
            std::chrono::duration_cast<MilliSecondsFP64>(endTime - startTime).count());

        ++passInd;
    }

    model->cleanUp();
}

//
// PassManager
//

#define ADD_DUMP_PASS(postfix) \
    passes->addPass(dumpModel(postfix), "dumpModel")

#define ADD_PASS(createFunc) \
    passes->addPass(createFunc(), # createFunc)

PassSet::Ptr PassManager::buildMiddleEnd() {
    const auto& env = CompileEnv::get();

    auto passes = std::make_shared<PassSet>();

    //
    // Initial state
    //

    _dumpInd = 0;
    ADD_DUMP_PASS("initial");
    ADD_PASS(addCopyForOutputsInsideNetwork);

    ADD_PASS(initialCheck);

    //
    // To overcome fp16 limitations
    //

    if (env.config.hwOptimization) {
        if (env.config.hwAdaptiveMode) {
            ADD_PASS(analyzeWeightableLayers);
        } else {
            if (!env.netConfig.hasManualDataScale()) {
                ADD_PASS(estimateSingleNetworkScale);
            }

            ADD_PASS(propagateDataScale);
        }

        ADD_DUMP_PASS("dataScaling");
    }

    //
    // Model common adaptation
    //

    ADD_PASS(removeUnusedStagesOutputs);
    ADD_DUMP_PASS("removeUnusedStagesOutputs");

    ADD_PASS(splitGroupedConv);
    ADD_DUMP_PASS("splitGroupedConv");

    //
    // Model HW-specific optimizations
    //

    if (env.config.hwOptimization) {
        ADD_PASS(replaceFCbyConv);
        ADD_DUMP_PASS("replaceFCbyConv");

        ADD_PASS(replaceDeconvByConv);
        ADD_DUMP_PASS("replaceDeconvByConv");

        if (env.config.hwDilation) {
            ADD_PASS(reshapeDilationConv);
            ADD_DUMP_PASS("reshapeDilationConv");
                }

        ADD_PASS(swapConcatAndHwOps);
        ADD_DUMP_PASS("swapConcatAndHwOps");

        ADD_PASS(mergeHwStages);
        ADD_DUMP_PASS("mergeHwStages");

        ADD_PASS(splitHwDepthConv);
        ADD_DUMP_PASS("splitHwDepthConv");

        ADD_PASS(splitHwConvAndPool);
        ADD_DUMP_PASS("splitHwConvAndPool");
    }

    ADD_PASS(hwPadding);
    ADD_DUMP_PASS("hwPadding");

    //
    // Batch support
    //

    ADD_PASS(adjustDataBatch);
    ADD_DUMP_PASS("adjustDataBatch");

    //
    // Replace StridedSlice to other stages
    //

    ADD_PASS(stridedSlice);
    ADD_DUMP_PASS("stridedSlice");

    //
    // HW stages tiling
    //

    if (env.config.hwOptimization) {
        ADD_PASS(hwConvTiling);
        ADD_PASS(hwPoolTiling);
        ADD_PASS(hwFullyConnectedTiling);
        ADD_DUMP_PASS("hwTiling");
    }

    //
    // Model SW-specific adaptation
    //

    ADD_PASS(swConvAdaptation);
    ADD_PASS(swDeconvAdaptation);
    ADD_PASS(swPoolAdaptation);
    ADD_PASS(swFullyConnectedAdaptation);
    ADD_DUMP_PASS("swAdaptation");

    //
    // Model SW-specific optimizations
    //

    ADD_PASS(mergeReLUAndBias);
    ADD_DUMP_PASS("mergeReLUAndBias");

    //
    // Data layout adjustment
    //

    ADD_PASS(adjustDataLayout);
    ADD_DUMP_PASS("adjustDataLayout");

    //
    // Model SW-specific optimizations after data layout adjustment
    //

    ADD_PASS(mergeEltwiseAndReLU);
    ADD_DUMP_PASS("mergeEltwiseAndReLU");

    //
    // Model special stages processing
    //

    ADD_PASS(processSpecialStages);
    ADD_DUMP_PASS("processSpecialStages");

    //
    // Data location adjustment
    //

    ADD_PASS(adjustDataLocation);
    ADD_DUMP_PASS("adjustDataLocation");

    //
    // Model common optimizations
    //

    if (env.config.copyOptimization.getOrDefault(true)) {
        ADD_PASS(eliminateCopyStages);
        ADD_DUMP_PASS("eliminateCopyStages");
    }

    //
    // HW/SW injection

    if (env.config.hwOptimization && env.config.injectSwOps.getOrDefault(true)) {
        ADD_PASS(injectSw);
        ADD_DUMP_PASS("injectSw");
    }

    //
    // Final resource allocation
    //

    ADD_PASS(allocateResources);
    ADD_DUMP_PASS("allocateResources");

    //
    // HW stages finalization
    //

    if (env.config.hwOptimization) {
        ADD_PASS(finalizeHwOps);
        ADD_DUMP_PASS("hwFinalization");
    }

    //
    // Final check
    //

    ADD_PASS(finalCheck);

    return passes;
}

#undef ADD_DUMP_PASS
#undef ADD_PASS

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
