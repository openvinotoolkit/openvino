// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <memory>
#include <algorithm>
#include <set>
#include <map>
#include <string>

#include <vpu/compile_env.hpp>
#include <vpu/utils/ie_helpers.hpp>

namespace vpu {

void FrontEnd::parseInputAndOutputData(const Model::Ptr& model) {
    VPU_PROFILE(parseInputAndOutputData);

    const auto& env = CompileEnv::get();

    auto layoutPreference = LayoutPreference::AUTO;
    if (env.config.hwOptimization ||
        env.config.forceLayout == ComputeLayout::NCHW ||
        env.config.forceLayout == ComputeLayout::NCDHW) {
        layoutPreference = LayoutPreference::ChannelMajor;  // CHW, NCHW, NCDHW
    } else {
        layoutPreference = LayoutPreference::ChannelMinor;  // HWC, NHWC, NDHWC
    }

    // TODO: InferenceEngine doesn't support 3D HWC.

    auto parseIOStrides = [&](const std::string& name, Data& data) {
        const auto& match = env.config.ioStrides.find(name);
        if (match == env.config.ioStrides.end()) {
            return;
        }

        const auto reqs = StridesRequirement::fixed(match->second, data->desc());
        data->updateRequiredStrides(reqs);
    };

    //
    // Parse network inputs
    //

    for (const auto& inputInfo : _ieNetworkParser.networkInputs) {
        auto netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        DataDesc vpuDesc(ieData->getTensorDesc());
        if (vpuDesc.numDims() >= 4) {
            if (LayoutPreference::ChannelMajor == layoutPreference) {
                if (vpuDesc.dimsOrder() == DimsOrder::NDHWC)
                    vpuDesc.moveDim(Dim::C, 3);
                if (vpuDesc.dimsOrder() == DimsOrder::NHWC)
                    vpuDesc.moveDim(Dim::C, 2);
            } else {
                vpuDesc.moveDim(Dim::C, 0);
            }
        }

        auto vpuData = model->addInputData(ieData->getName(), vpuDesc);
        parseIOStrides(inputInfo.first, vpuData);

        bindData(vpuData, ieData);
    }

    model->attrs().set<int>("numInputs", _ieNetworkParser.networkInputs.size());

    //
    // Parse network outputs
    //

    for (const auto& outputInfo : _ieNetworkParser.networkOutputs) {
        auto ieData = outputInfo.second;
        IE_ASSERT(ieData != nullptr);

        DataDesc vpuDesc(ieData->getTensorDesc());
        if (vpuDesc.numDims() >= 4) {
            if (LayoutPreference::ChannelMajor == layoutPreference) {
                if (vpuDesc.dimsOrder() == DimsOrder::NDHWC)
                    vpuDesc.moveDim(Dim::C, 3);
                if (vpuDesc.dimsOrder() == DimsOrder::NHWC)
                    vpuDesc.moveDim(Dim::C, 2);
            } else {
                vpuDesc.moveDim(Dim::C, 0);
            }
        }

        auto vpuData = model->addOutputData(ieData->getName(), vpuDesc);
        parseIOStrides(outputInfo.first, vpuData);

        bindData(vpuData, ieData);

        if (_unbatchedOutputs.count(ieData) > 0) {
            vpuData->attrs().set<bool>("unbatched", true);
        }
    }

    model->attrs().set<int>("numOutputs", _ieNetworkParser.networkOutputs.size());

    //
    // Parse constant data
    //

    for (const auto& constInfo : _ieNetworkParser.constDatas) {
        auto ieData = constInfo.first;
        IE_ASSERT(ieData != nullptr);

        auto ieBlob = constInfo.second;
        IE_ASSERT(ieBlob != nullptr);

        DataDesc vpuDesc(ieData->getTensorDesc());

        auto vpuData = model->addConstData(
            ieData->getName(),
            vpuDesc,
            ieBlobContent(ieBlob));

        // User might ask to return the output from Const layer.
        if (auto vpuOutData = getVpuData(ieData)) {
            IE_ASSERT(vpuOutData->usage() == DataUsage::Output);

            _stageBuilder->addCopyStage(
                model,
                formatString("%s@return-const", vpuData->name()),
                nullptr,
                vpuData,
                vpuOutData);
        }

        bindData(vpuData, ieData);
    }
}

}  // namespace vpu
