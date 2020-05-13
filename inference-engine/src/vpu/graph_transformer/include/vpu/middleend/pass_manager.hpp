// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <vpu/model/model.hpp>
#include <vpu/stage_builder.hpp>
#include <vpu/backend/backend.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

//
// Pass
//

class Pass {
public:
    using Ptr = std::shared_ptr<Pass>;

    virtual ~Pass() = default;

    virtual void run(const Model& model) = 0;
};

//
// PerStagePass
//

class PerStagePass : public Pass {
public:
    explicit PerStagePass(std::initializer_list<StageType> types) : _types(types) {}

    void run(const Model& model) override;

protected:
    virtual void runForStage(const Model& model, const Stage& stage) = 0;

private:
    EnumSet<StageType> _types;
};

//
// PassSet
//

class PassSet final {
public:
    using Ptr = std::shared_ptr<PassSet>;

    void run(const Model& model) const;

    inline void addPass(
            const Pass::Ptr& pass,
            const std::string& name = std::string()) {
        _passes.emplace_back(pass, name);
    }

private:
    std::vector<std::pair<Pass::Ptr, std::string>> _passes;
};

//
// PassManager
//

class PassManager final {
public:
    using Ptr = std::shared_ptr<PassManager>;

    PassManager(
            const StageBuilder::Ptr& stageBuilder,
            const BackEnd::Ptr& backEnd) :
            _stageBuilder(stageBuilder), _backEnd(backEnd) {
    }

    PassSet::Ptr buildMiddleEnd();

public:
    //
    // To overcome fp16 limitations
    //

    Pass::Ptr analyzeWeightableLayers();

    //
    // Model common adaptation
    //

    Pass::Ptr eliminateConstConcat();
    Pass::Ptr splitGroupedConv();
    Pass::Ptr splitConv3DInto2D();
    Pass::Ptr splitPool3DInto2D();
    Pass::Ptr eliminateRedundantConversions();

    //
    // Model HW-specific optimizations
    //

    Pass::Ptr replaceFCbyConv();
    Pass::Ptr replaceDeconvByConv();
    Pass::Ptr swapConcatAndHwOps();
    Pass::Ptr mergeHwStages();
    Pass::Ptr splitHwDepthConv();
    Pass::Ptr splitHwConvAndPool();
    Pass::Ptr hwPadding();
    Pass::Ptr splitLargeKernelConv();

    //
    // Batch support
    //

    Pass::Ptr adjustDataBatch();

    //
    // HW stages tiling
    //

    Pass::Ptr hwConvTiling();
    Pass::Ptr hwPoolTiling();
    Pass::Ptr hwFullyConnectedTiling();
    Pass::Ptr hwExtraSplit();

    //
    // Model SW-specific adaptation
    //

    Pass::Ptr swConvAdaptation();
    Pass::Ptr swDeconvAdaptation();
    Pass::Ptr swPoolAdaptation();
    Pass::Ptr swFullyConnectedAdaptation();

    //
    // Model SW-specific optimizations
    //

    Pass::Ptr mergeReLUAndBias();
    Pass::Ptr mergeEltwiseAndReLU();
    Pass::Ptr replaceWithSCReLU();
    Pass::Ptr replaceWithReduceMean();

    //
    // StridedSlice processing
    //

    Pass::Ptr stridedSlice();

    //
    // PriorBox[Clustered] replacing
    //

    Pass::Ptr replacePriorBoxWithConst();

    //
    // Data layout adjustment
    //

    Pass::Ptr adjustDataLayout();

    //
    // Model special stages processing
    //

    Pass::Ptr processSpecialStages();

    //
    // Data location adjustment
    //

    Pass::Ptr adjustDataLocation();

    //
    // Model common optimizations
    //

    Pass::Ptr eliminateCopyStages();
    Pass::Ptr mergePermuteStages();
    Pass::Ptr upliftActivationStages();

    //
    // HW/SW injection
    //

    Pass::Ptr injectSw();

    //
    // Final resource allocation
    //

    Pass::Ptr allocateResources();

    //
    // HW stages finalization
    //

    Pass::Ptr finalizeHwOps();

    //
    // Final check
    //

    Pass::Ptr finalCheck();

    //
    // Debug passes
    //

    Pass::Ptr dumpModel(const std::string& postfix);

    //
    // Dilation Conv NCE  passes
    //

    Pass::Ptr reshapeDilationConv();

    Pass::Ptr addCopyForOutputsInsideNetwork();

    Pass::Ptr initialCheck();

    //
    // Reorder input for Myriad2
    //

    Pass::Ptr reorderInputsToChannelMinor();

    Pass::Ptr mergeParallelFC();

    Pass::Ptr gemmTranspose();

    Pass::Ptr countStagesInLoops();

    Pass::Ptr replaceGemmByConv();

protected:
    StageBuilder::Ptr _stageBuilder;
    BackEnd::Ptr _backEnd;

    int _dumpInd = 0;
};

}  // namespace vpu
