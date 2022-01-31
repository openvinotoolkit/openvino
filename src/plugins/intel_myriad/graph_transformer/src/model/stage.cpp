// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/stage.hpp>

#include <queue>
#include <algorithm>
#include <vector>
#include <string>

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

        const auto totalNumSHAVEs = model()->attrs().get<Resources>("resources").numSHAVEs;
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

    if (const auto injectedStage = this->injectedStage()) {
        injectedStage->_numSHAVEs = _numSHAVEs;
    }
}

const StageDataInfo<DimsOrder>& StageNode::propagateDataOrder() {
    //
    // Get result from Stage implementation.
    //

    _orderInfo.init(_inputEdges.size(), _outputEdges.size());
    propagateDataOrderImpl(_orderInfo);

    //
    // Merge with the results from injected Stages.
    //

    if (const auto injectedStage = this->injectedStage()) {
        const auto& childRes = injectedStage->propagateDataOrder();

        for (const auto& inEdge : injectedStage->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_orderInfo.hasInput(inEdge->parentEdge()));
                _orderInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : injectedStage->outputEdges()) {
            if (childRes.hasOutput(outEdge)) {
                IE_ASSERT(!_orderInfo.hasOutput(outEdge->parentEdge()));
                _orderInfo.setOutput(outEdge->parentEdge(), childRes.getOutput(outEdge));
            }
        }
    }

    return _orderInfo;
}

const StageDataInfo<StridesRequirement>& StageNode::getDataStridesRequirements() {
    //
    // Get result from Stage implementation.
    //

    _stridesInfo.init(_inputEdges.size(), _outputEdges.size());
    getDataStridesRequirementsImpl(_stridesInfo);

    //
    // Merge with the results from injected Stages.
    //

    if (const auto injectedStage = this->injectedStage()) {
        const auto& childRes = injectedStage->getDataStridesRequirements();

        for (const auto& inEdge : injectedStage->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_stridesInfo.hasInput(inEdge->parentEdge()));
                _stridesInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : injectedStage->outputEdges()) {
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

    VPU_INTERNAL_CHECK(
        parentStageEdge() == nullptr && injectedStageEdge() == nullptr,
        "finalizeDataLayout was called for Stage node %v which is a part of Injection pair", this);

    finalizeDataLayoutImpl();
}

const StageDataInfo<BatchSupport>& StageNode::getBatchSupportInfo() {
    //
    // Get result from Stage implementation.
    //

    _batchInfo.init(_inputEdges.size(), _outputEdges.size());
    getBatchSupportInfoImpl(_batchInfo);

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

    if (const auto injectedStage = this->injectedStage()) {
        const auto& childRes = injectedStage->getBatchSupportInfo();

        for (const auto& inEdge : injectedStage->inputEdges()) {
            if (childRes.hasInput(inEdge)) {
                IE_ASSERT(!_batchInfo.hasInput(inEdge->parentEdge()));
                _batchInfo.setInput(inEdge->parentEdge(), childRes.getInput(inEdge));
            }
        }
        for (const auto& outEdge : injectedStage->outputEdges()) {
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

    auto reqs = getSHAVEsRequirementsImpl();

    //
    // Merge with the results from injected Stages.
    //

    if (const auto injectedStage = this->injectedStage()) {
        auto childRes = injectedStage->getSHAVEsRequirements();

        auto resVal = static_cast<int>(reqs);
        auto childResVal = static_cast<int>(childRes);

        reqs = static_cast<StageSHAVEsRequirements>(std::max(resVal, childResVal));
    }

    return reqs;
}

void StageNode::initialCheck() const {
    try {
        initialCheckImpl();
    } catch (const InferenceEngine::Exception& exception) {
        VPU_THROW_EXCEPTION << name() << " of type " << type() << ": " << exception.what();
    }

    if (const auto injectedStage = this->injectedStage()) {
        try {
            injectedStage->initialCheck();
        } catch (const InferenceEngine::Exception& exception) {
            VPU_THROW_EXCEPTION << name() << " of type " << type() << ": " << exception.what();
        }
    }
}

void StageNode::finalCheck() const {
    try {
        finalCheckImpl();
    } catch (const InferenceEngine::Exception& exception) {
        VPU_THROW_EXCEPTION << name() << " of type " << type() << ": " << exception.what();
    }

    if (const auto injectedStage = this->injectedStage()) {
        try {
            injectedStage->finalCheck();
        } catch (const ie::Exception& exception) {
            VPU_THROW_EXCEPTION << name() << " of type " << type() << ": " << exception.what();
        }
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

StageSHAVEsRequirements StageNode::getSHAVEsRequirementsImpl() const {
    if (category() == StageCategory::SHAVE) {
        return StageSHAVEsRequirements::NeedMax;
    } else {
        return StageSHAVEsRequirements::NotNeeded;
    }
}

void printTo(std::ostream& os, const Stage& stage) {
    if (stage == nullptr) {
        os << "<null>";
    } else {
        os << stage->name() << " (" << stage->type() << ")";
    }
}

void assertAllInputsOutputsTypes(const Stage& stage,
                                 const DataType& expectedInputsType,
                                 const DataType& expectedOutputsType) {
    auto assertTypes = [stage](const DataType& expectedType,
                               const std::vector<Data>& datas,
                               const std::string& token) {
        for (decltype(datas.size()) idx = 0; idx < datas.size(); ++idx) {
            if (datas[idx]->usage() == DataUsage::Fake)
                continue;

            const auto& actualType = datas[idx]->desc().type();

            VPU_THROW_UNLESS(
                actualType == expectedType,
                "Stage node %v types check error: %v #%v has type %v, but %v is expected",
                stage, token, idx, actualType, expectedType);
        }
    };

    assertTypes(expectedInputsType, toVector(stage->inputs()), "input");
    assertTypes(expectedOutputsType, toVector(stage->outputs()), "output");
}

void assertInputsOutputsTypes(const Stage& stage,
                              const DataTypesRequirement& expectedInputsTypes,
                              const DataTypesRequirement& expectedOutputsTypes) {
    auto assertTypes = [stage](const DataTypesRequirement& expectedTypes,
                               const std::vector<Data>& datas,
                               const std::string& token) {
        VPU_THROW_UNLESS(
            expectedTypes.size() == datas.size(),
            "Stage node %v types check error: the Stage has %v of %v edges, but %v is expected",
            stage, datas.size(), token, expectedTypes.size());

        for (decltype(datas.size()) idx = 0; idx < datas.size(); ++idx) {
            if (datas[idx]->usage() == DataUsage::Fake)
                continue;

            const auto& possibleTypes = expectedTypes[idx];
            const auto& actualType = datas[idx]->desc().type();

            VPU_THROW_UNLESS(
                possibleTypes.find(actualType) != possibleTypes.end(),
                "Stage node %v types check error: %v #%v has type %v, but one of %v is expected",
                stage, token, idx, actualType, possibleTypes);
        }
    };

    assertTypes(expectedInputsTypes, toVector(stage->inputs()), "input");
    assertTypes(expectedOutputsTypes, toVector(stage->outputs()), "output");
}

}  // namespace vpu
