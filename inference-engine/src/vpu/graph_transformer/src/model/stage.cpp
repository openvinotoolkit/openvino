// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/stage.hpp>

#include <queue>
#include <algorithm>
#include <vector>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/model.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

void StageNode::setNumSHAVEs(int numSHAVEs) {
    if (_parentStageEdge == nullptr) {
        //
        // Check resources assigned to current Model.
        //

        IE_ASSERT(_model != nullptr);

        auto totalNumSHAVEs = _model->attrs().get<Resources>("resources").numSHAVEs;
        IE_ASSERT(numSHAVEs <= totalNumSHAVEs);
    } else {
        //
        // Check resources assigned to parent stage.
        //

        IE_ASSERT(numSHAVEs == _parentStageEdge->parent()->_numSHAVEs);
    }

    _numSHAVEs =  numSHAVEs;

    //
    // Propagate SHAVEs to injected children.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        injectedStageEdge->child()->_numSHAVEs = _numSHAVEs;
    }
}

const StageDataInfo<float>& StageNode::propagateScaleFactors(
        const SmallVector<float>& inputScales,
        ScalePropagationStep step) {
    //
    // Stage <-> Stage edges are not allowed here.
    //

    IE_ASSERT(_parentStageEdge == nullptr);
    IE_ASSERT(_injectedStageEdges.empty());

    //
    // Check that `inputScales` is valid.
    //

    IE_ASSERT(inputScales.size() == _inputEdges.size());

    //
    // Get result from Stage implementation.
    //

    _scaleInfo.init(_inputEdges.size(), _outputEdges.size());
    propagateScaleFactorsImpl(inputScales, step);

    //
    // Check that implementation returned valid map.
    //

#ifndef NDEBUG
    for (const auto& outEdge : _outputEdges) {
        IE_ASSERT(_scaleInfo.hasOutput(outEdge));
    }
#endif

    return _scaleInfo;
}

const StageDataInfo<DimsOrder>& StageNode::propagateDataOrder() const {
    //
    // Get result from Stage implementation.
    //

    _orderInfo.init(_inputEdges.size(), _outputEdges.size());
    propagateDataOrderImpl();

    //
    // Merge with the results from injected Stages.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        const auto& child = injectedStageEdge->child();
        const auto& childRes = child->propagateDataOrder();

        for (const auto& inEdge : child->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_orderInfo.hasInput(inEdge->parentEdge()));
                _orderInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : child->outputEdges()) {
            if (childRes.hasOutput(outEdge)) {
                IE_ASSERT(!_orderInfo.hasOutput(outEdge->parentEdge()));
                _orderInfo.setOutput(outEdge->parentEdge(), childRes.getOutput(outEdge));
            }
        }
    }

    return _orderInfo;
}

const StageDataInfo<StridesRequirement>& StageNode::getDataStridesRequirements() const {
    //
    // Get result from Stage implementation.
    //

    _stridesInfo.init(_inputEdges.size(), _outputEdges.size());
    getDataStridesRequirementsImpl();

    //
    // Merge with the results from injected Stages.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        const auto& child = injectedStageEdge->child();
        const auto& childRes = child->getDataStridesRequirements();

        for (const auto& inEdge : child->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_stridesInfo.hasInput(inEdge->parentEdge()));
                _stridesInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : child->outputEdges()) {
            if (childRes.hasOutput(outEdge)) {
                IE_ASSERT(!_stridesInfo.hasOutput(outEdge->parentEdge()));
                _stridesInfo.setOutput(outEdge->parentEdge(), childRes.getOutput(outEdge));
            }
        }
    }

    return _stridesInfo;
}

void StageNode::finalizeDataLayout() {
    //
    // Stage <-> Stage edges are not allowed here.
    //

    IE_ASSERT(_parentStageEdge == nullptr);
    IE_ASSERT(_injectedStageEdges.empty());

    finalizeDataLayoutImpl();
}

const StageDataInfo<BatchSupport>& StageNode::getBatchSupportInfo() const {
    //
    // Get result from Stage implementation.
    //

    _batchInfo.init(_inputEdges.size(), _outputEdges.size());
    getBatchSupportInfoImpl();

    //
    // Check that implemenation returned valid map.
    //

#ifndef NDEBUG
    bool hasSplit = false;
    for (const auto& inEdge : _inputEdges) {
        if (inEdge->childEdge() != nullptr) {
            continue;
        }

        if (_batchInfo.hasInput(inEdge)) {
            auto curReq = _batchInfo.getInput(inEdge);

            if (curReq == BatchSupport::Split) {
                hasSplit = true;
            } else {
                IE_ASSERT(curReq == BatchSupport::ReplicateConstContent);
            }
        }
    }

    for (const auto& outEdge : _outputEdges) {
        if (outEdge->childEdge() != nullptr) {
            continue;
        }

        if (hasSplit) {
            IE_ASSERT(_batchInfo.hasOutput(outEdge));

            auto curReq = _batchInfo.getOutput(outEdge);
            IE_ASSERT(curReq == BatchSupport::Split);
        } else {
            IE_ASSERT(!_batchInfo.hasOutput(outEdge));
        }
    }
#endif

    //
    // Merge with the results from injected Stages.
    //

    //
    // Do this after the checks, because parent and child Stages might have different requirements.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        const auto& child = injectedStageEdge->child();
        const auto& childRes = child->getBatchSupportInfo();

        for (const auto& inEdge : child->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_batchInfo.hasInput(inEdge->parentEdge()));
                _batchInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : child->outputEdges()) {
            if (childRes.hasOutput(outEdge)) {
                IE_ASSERT(!_batchInfo.hasOutput(outEdge->parentEdge()));
                _batchInfo.setOutput(outEdge->parentEdge(), childRes.getOutput(outEdge));
            }
        }
    }

    return _batchInfo;
}

StageSHAVEsRequirements StageNode::getSHAVEsRequirements() const {
    //
    // Get result from Stage implementation.
    //

    // return max for Myriad2
    const auto& compileEnv = CompileEnv::get();
    if (compileEnv.platform == Platform::MYRIAD_2) {
        return StageSHAVEsRequirements::NeedMax;
    }

    auto reqs = getSHAVEsRequirementsImpl();

    //
    // Merge with the results from injected Stages.
    //

    for (const auto& injectedStageEdge : injectedStageEdges()) {
        auto childRes = injectedStageEdge->child()->getSHAVEsRequirements();

        auto resVal = static_cast<int>(reqs);
        auto childResVal = static_cast<int>(childRes);

        reqs = static_cast<StageSHAVEsRequirements>(std::max(resVal, childResVal));
    }

    return reqs;
}

void StageNode::finalCheck() const {
    finalCheckImpl();

    for (const auto& injectedStageEdge : injectedStageEdges()) {
        injectedStageEdge->child()->finalCheck();
    }
}

void StageNode::serialize(BlobSerializer& serializer) const {
    // Check that we don't serialize Special stage.
    IE_ASSERT(category() != StageCategory::Special);

    mv_stage_header stageHdr = {
        checked_cast<uint32_t>(0u),
        checked_cast<uint32_t>(_type),
        checked_cast<uint32_t>(_numSHAVEs)
    };

    auto stageHeaderPos = serializer.append(stageHdr);

    auto paramsPos = serializer.append(static_cast<uint32_t>(0));
    serializeParamsImpl(serializer);
    serializer.overWriteTailSize(paramsPos);

    serializeDataImpl(serializer);

    serializer.append(stageHdr.stage_type);
    serializer.append(STAGE_BORDER_SYMBOL);

    serializer.overWriteTailSize(stageHeaderPos);
}

void StageNode::propagateScaleFactorsImpl(
        const SmallVector<float>&,
        ScalePropagationStep) {
    //
    // Default implementation assumes no scaling support.
    //

    for (const auto& inEdge : _inputEdges) {
        _scaleInfo.setInput(inEdge, 1.0f);
    }
    for (const auto& outEdge : _outputEdges) {
        _scaleInfo.setOutput(outEdge, 1.0f);
    }
}

StageSHAVEsRequirements StageNode::getSHAVEsRequirementsImpl() const {
    if (category() == StageCategory::SHAVE) {
        return StageSHAVEsRequirements::NeedMax;
    } else {
        return StageSHAVEsRequirements::NotNeeded;
    }
}

void printTo(std::ostream& os, const Stage& stage) {
    os << (stage == nullptr ? "<null>" : stage->name());
}

}  // namespace vpu
