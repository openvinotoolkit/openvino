// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include <vpu/model/model.hpp>
#include <vpu/frontend/stage_builder.hpp>
#include <vpu/backend/backend.hpp>

namespace vpu {

//
// Pass
//

class Pass : public std::enable_shared_from_this<Pass> {
public:
    using Ptr = std::shared_ptr<Pass>;

    virtual ~Pass() = default;

    virtual void run(const Model::Ptr& model) = 0;
};

//
// PerStagePass
//

class PerStagePass : public Pass {
public:
    explicit PerStagePass(std::initializer_list<StageType> types) : _types(types) {}

    void run(const Model::Ptr& model) override;

protected:
    virtual void runForStage(const Model::Ptr& model, const Stage& stage) = 0;

private:
    EnumSet<StageType> _types;
};

//
// PassSet
//

class PassSet final : public std::enable_shared_from_this<PassSet> {
public:
    using Ptr = std::shared_ptr<PassSet>;

    void run(const Model::Ptr& model) const;

    void addPass(const Pass::Ptr& pass) { _passes.emplace_back(pass); }
    void addPass(Pass::Ptr&& pass) { _passes.emplace_back(std::move(pass)); }

private:
    std::vector<Pass::Ptr> _passes;
};

//
// PassManager
//

class PassManager final : public std::enable_shared_from_this<PassManager> {
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
    // Find SubGraphs for allocation
    //

    Pass::Ptr findSubGraphs();

    //
    // To overcome fp16 limitations
    //

    Pass::Ptr analyzeWeightableLayers();
    Pass::Ptr estimateSingleNetworkScale();
    Pass::Ptr propagateDataScale();

    //
    // Model common adaptation
    //

    Pass::Ptr splitGroupedConv();

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

protected:
    StageBuilder::Ptr _stageBuilder;
    BackEnd::Ptr _backEnd;

    int _dumpInd = 0;
};

}  // namespace vpu
