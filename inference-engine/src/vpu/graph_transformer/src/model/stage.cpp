// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/stage.hpp>

#include <queue>
#include <algorithm>

#include <vpu/model/edges.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/model.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

void StageNode::setSubGraphNumber(int subGraphNumber) {
    IE_ASSERT(subGraphNumber >= -1);
    _subGraphNumber = subGraphNumber;
}

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

DataMap<float> StageNode::propagateScaleFactors(
        const DataMap<float>& inputScales,
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
    for (const auto& inEdge : _inputEdges) {
        IE_ASSERT(inputScales.count(inEdge->input()) > 0);
    }

    //
    // Get result from Stage implementation.
    //

    auto res = propagateScaleFactorsImpl(inputScales, step);

    //
    // Check that implementation returned valid map.
    //

#ifndef NDEBUG
    IE_ASSERT(res.size() <= (_inputEdges.size() + _outputEdges.size()));

    for (const auto& p : res) {
        auto it1 = std::find_if(_inputEdges.begin(), _inputEdges.end(), [p](const StageInput& inEdge) {
            return inEdge->input() == p.first;
        });
        auto it2 = std::find_if(_outputEdges.begin(), _outputEdges.end(), [p](const StageOutput& outEdge) {
            return outEdge->output() == p.first;
        });
        IE_ASSERT(it1 != _inputEdges.end() || it2 != _outputEdges.end());
    }

    for (const auto& outEdge : _outputEdges) {
        IE_ASSERT(res.count(outEdge->output()) > 0);
    }
#endif

    return res;
}

DataMap<DimsOrder> StageNode::propagateDataOrder() const {
    //
    // Get result from Stage implementation.
    //

    auto res = propagateDataOrderImpl();

    //
    // Merge with the results from injected Stages.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        auto childRes = injectedStageEdge->child()->propagateDataOrder();
        res.insert(childRes.begin(), childRes.end());
    }

    //
    // Check that implemenation returned valid map.
    //

#ifndef NDEBUG
    IE_ASSERT(res.size() <= (_inputEdges.size() + _outputEdges.size()));

    for (const auto& p : res) {
        auto it1 = std::find_if(_inputEdges.begin(), _inputEdges.end(), [p](const StageInput& inEdge) {
            return inEdge->input() == p.first;
        });
        auto it2 = std::find_if(_outputEdges.begin(), _outputEdges.end(), [p](const StageOutput& outEdge) {
            return outEdge->output() == p.first;
        });
        IE_ASSERT(it1 != _inputEdges.end() || it2 != _outputEdges.end());
    }
#endif

    return res;
}

DataMap<StridesRequirement> StageNode::getDataStridesRequirements() const {
    //
    // Get result from Stage implementation.
    //

    auto res = getDataStridesRequirementsImpl();

    //
    // Merge with the results from injected Stages.
    //

    for (const auto& injectedStageEdge : _injectedStageEdges) {
        auto childRes = injectedStageEdge->child()->getDataStridesRequirements();
        res.insert(childRes.begin(), childRes.end());
    }

    //
    // Check that implemenation returned valid map.
    //

#ifndef NDEBUG
    IE_ASSERT(res.size() <= (_inputEdges.size() + _outputEdges.size()));

    for (const auto& p : res) {
        auto it1 = std::find_if(_inputEdges.begin(), _inputEdges.end(), [p](const StageInput& inEdge) {
            return inEdge->input() == p.first;
        });
        auto it2 = std::find_if(_outputEdges.begin(), _outputEdges.end(), [p](const StageOutput& outEdge) {
            return outEdge->output() == p.first;
        });
        IE_ASSERT(it1 != _inputEdges.end() || it2 != _outputEdges.end());
    }
#endif

    return res;
}

void StageNode::finalizeDataLayout() {
    //
    // Stage <-> Stage edges are not allowed here.
    //

    IE_ASSERT(_parentStageEdge == nullptr);
    IE_ASSERT(_injectedStageEdges.empty());

    finalizeDataLayoutImpl();
}

DataMap<BatchSupport> StageNode::getBatchSupportInfo() const {
    //
    // Get result from Stage implementation.
    //

    auto res = getBatchSupportInfoImpl();

    //
    // Check that implemenation returned valid map.
    //

#ifndef NDEBUG
    IE_ASSERT(res.size() <= (_inputEdges.size() + _outputEdges.size()));

    for (const auto& p : res) {
        auto it1 = std::find_if(_inputEdges.begin(), _inputEdges.end(), [p](const StageInput& inEdge) {
            return inEdge->input() == p.first;
        });
        auto it2 = std::find_if(_outputEdges.begin(), _outputEdges.end(), [p](const StageOutput& outEdge) {
            return outEdge->output() == p.first;
        });
        IE_ASSERT(it1 != _inputEdges.end() || it2 != _outputEdges.end());
    }

    bool hasSplit = false;
    for (const auto& inEdge : _inputEdges) {
        if (inEdge->childEdge() != nullptr) {
            continue;
        }

        auto input = inEdge->input();

        auto it = res.find(input);
        if (it != res.end()) {
            auto curReq = it->second;

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

        auto it = res.find(outEdge->output());
        if (hasSplit) {
            IE_ASSERT(it != res.end());

            auto curReq = it->second;
            IE_ASSERT(curReq == BatchSupport::Split);
        } else {
            IE_ASSERT(it == res.end());
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
        auto childRes = injectedStageEdge->child()->getBatchSupportInfo();
        res.insert(childRes.begin(), childRes.end());
    }

    return res;
}

StageSHAVEsRequirements StageNode::getSHAVEsRequirements() const {
    //
    // Get result from Stage implementation.
    //

    // return max for Myriad2
    auto compileEnv = CompileEnv::get();
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

DataMap<float> StageNode::propagateScaleFactorsImpl(
        const DataMap<float>&,
        ScalePropagationStep) {
    //
    // Default implementation assumes no scaling support.
    //

    DataMap<float> out;

    for (const auto& inEdge : _inputEdges) {
        out[inEdge->input()] = 1.0f;
    }
    for (const auto& outEdge : _outputEdges) {
        out[outEdge->output()] = 1.0f;
    }

    return out;
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
