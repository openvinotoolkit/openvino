// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/pass_manager.hpp>

#include <sstream>
#include <iomanip>
#include <memory>
#include <string>

#include <vpu/compile_env.hpp>

namespace vpu {

//
// PerStagePass
//

void PerStagePass::run(const Model& model) {
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

void PassSet::run(const Model& model) const {
    using MilliSecondsFP64 = std::chrono::duration<double, std::milli>;

    const auto& env = CompileEnv::get();

    env.log->debug("MiddleEnd : Run passes");
    VPU_LOGGER_SECTION(env.log);

    int passInd = 0;
    for (const auto& p : _passes) {
        env.log->debug("Start pass %m%d / %d [%s]", std::setw(2), passInd + 1, _passes.size(), p.second);
        VPU_LOGGER_SECTION(env.log);

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

    // initial dump pass must be the first dump
    ADD_DUMP_PASS("initial");

    if (!env.config.disableReorder && !env.config.hwOptimization) {
        ADD_PASS(reorderInputsToChannelMinor);
        ADD_DUMP_PASS("reorderInputsToChannelMinor");
    }

    ADD_PASS(addCopyForOutputsInsideNetwork);
    ADD_DUMP_PASS("addCopyForOutputsInsideNetwork");

    ADD_PASS(initialCheck);

    //
    // Replace PriorBox[Clustered] with ConstData
    //

    ADD_PASS(replacePriorBoxWithConst);
    ADD_DUMP_PASS("replacePriorBoxWithConst");

    //
    // 3D layers adaptation
    // (do this before `analyzeWeightableLayers`)
    //

    ADD_PASS(splitConv3DInto2D);
    ADD_DUMP_PASS("splitConv3DInto2D");

    ADD_PASS(splitPool3DInto2D);
    ADD_DUMP_PASS("splitPool3DInto2D");

    //
    // To overcome fp16 limitations
    //

    if (env.config.hwOptimization) {
        ADD_PASS(analyzeWeightableLayers);
        ADD_DUMP_PASS("analyzeWeightableLayers");
    }

    ADD_PASS(mergeParallelFC);
    ADD_DUMP_PASS("mergeParallelFC");

    //
    // Replace Global AvgPooling with ReduceMean
    //

    if (env.config.enableReplaceWithReduceMean) {
        ADD_PASS(replaceWithReduceMean);
        ADD_DUMP_PASS("replaceWithReduceMean");
    }

    //
    // Model common adaptation
    //

    ADD_PASS(eliminateConstConcat);
    ADD_DUMP_PASS("eliminateConstConcat");

    ADD_PASS(splitGroupedConv);
    ADD_DUMP_PASS("splitGroupedConv");

    ADD_PASS(eliminateRedundantConversions);
    ADD_DUMP_PASS("eliminateRedundantConversions");

    //
    // Model HW-specific optimizations
    //

    if (env.config.hwOptimization) {
        ADD_PASS(replaceFCbyConv);
        ADD_DUMP_PASS("replaceFCbyConv");

        // TODO: enable this pass after Permute optimization
        // ADD_PASS(replaceGemmByConv);
        // ADD_DUMP_PASS("replaceGemmByConv");

        ADD_PASS(replaceDeconvByConv);
        ADD_DUMP_PASS("replaceDeconvByConv");

        if (env.config.hwDilation) {
            ADD_PASS(reshapeDilationConv);
            ADD_DUMP_PASS("reshapeDilationConv");
        }

        ADD_PASS(upliftActivationStages);
        ADD_DUMP_PASS("upliftActivationStages");

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

    if (env.config.enableReplWithSCRelu) {
        ADD_PASS(replaceWithSCReLU);
        ADD_DUMP_PASS("replaceWithSCReLU");
    }

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

        if (env.config.hwExtraSplit) {
            ADD_PASS(hwExtraSplit);
            ADD_DUMP_PASS("hwExtraSplit");
        }
    }

    //
    // Model SW-specific adaptation
    //

    ADD_PASS(swConvAdaptation);
    ADD_PASS(swDeconvAdaptation);
    ADD_PASS(swPoolAdaptation);
    ADD_PASS(swFullyConnectedAdaptation);
    ADD_DUMP_PASS("swAdaptation");

    ADD_PASS(gemmTranspose);
    ADD_DUMP_PASS("gemmTranspose");

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
    // Model common optimizations after data layout adjustment
    //

    // TODO: mergePermute support for reorder stage too.
    // TODO: pass that will swap Permute and per-element operations.
    if (env.config.enablePermuteMerging) {
        ADD_PASS(mergePermuteStages);
        ADD_DUMP_PASS("mergePermuteStages");
    }

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

    ADD_PASS(countStagesInLoops);
    ADD_DUMP_PASS("countStagesInLoops");

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

    void run(const Model& model) override {
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
