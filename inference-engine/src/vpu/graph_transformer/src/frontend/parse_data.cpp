// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vpu/configuration/options/tensor_strides.hpp>

#include <memory>
#include <algorithm>
#include <set>
#include <map>
#include <string>

namespace vpu {

void FrontEnd::parseInputAndOutputData(const Model& model) {
    VPU_PROFILE(parseInputAndOutputData);

    const auto& env = CompileEnv::get();

    env.log->trace("Parse input and output data");
    VPU_LOGGER_SECTION(env.log);

    const auto parseIOStrides = [&env](const std::string& name, const Data& data) {
        const auto tensorStrides = env.config.get<TensorStridesOption>();
        const auto& match = tensorStrides.find(name);
        if (match == tensorStrides.end()) {
            return;
        }

        env.log->trace("Data %s has fixed strides %v", data, match->second);
        VPU_LOGGER_SECTION(env.log);

        const auto reqs = StridesRequirement::fixed(match->second, data->desc());
        data->updateRequiredStrides(reqs);
    };

    model->attrs().set<int>("numInputs", checked_cast<int>(_ieParsedNetwork.networkInputs.size()));
    model->attrs().set<int>("numOutputs", checked_cast<int>(_ieParsedNetwork.networkOutputs.size()));

    //
    // Parse network inputs
    //

    env.log->trace("Parse network inputs");

    for (const auto& inputInfo : _ieParsedNetwork.networkInputs) {
        const auto& netInput = inputInfo.second;
        IE_ASSERT(netInput != nullptr);

        const auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        env.log->trace("Network input : %s", ieData->getName());
        VPU_LOGGER_SECTION(env.log);

        const auto vpuDesc = DataDesc{ieData->getTensorDesc()};
        const auto vpuData = model->addInputData(ieData->getName(), vpuDesc);

        parseIOStrides(inputInfo.first, vpuData);

        bindData(vpuData, ieData);
    }

    //
    // Parse network outputs
    //

    env.log->trace("Parse network outputs");

    for (const auto& outputInfo : _ieParsedNetwork.networkOutputs) {
        const auto& ieData = outputInfo.second;
        IE_ASSERT(ieData != nullptr);

        env.log->trace("Network output : %s", ieData->getName());
        VPU_LOGGER_SECTION(env.log);

        const auto vpuDesc = DataDesc{ieData->getTensorDesc()};
        const auto vpuData = model->addOutputData(ieData->getName(), vpuDesc);

        parseIOStrides(outputInfo.first, vpuData);

        bindData(vpuData, ieData);

        if (_unbatchedOutputs.count(ieData) > 0) {
            env.log->trace("The output %s is unbatched", vpuData);
            vpuData->attrs().set<bool>("unbatched", true);
        }
    }

    //
    // Parse network constant
    //

    env.log->trace("Parse network constants");

    for (const auto& constInfo : _ieParsedNetwork.constDatas) {
        const auto& ieData = constInfo.first;
        IE_ASSERT(ieData != nullptr);

        env.log->trace("Network constant : %s", ieData->getName());
        VPU_LOGGER_SECTION(env.log);

        const auto& ieBlob = constInfo.second;
        IE_ASSERT(ieBlob != nullptr);

        auto descriptor = DataDesc{ieData->getTensorDesc()};
        if (descriptor.type() == DataType::FP32) {
            descriptor.setType(DataType::FP16);
        }

        const auto vpuData = model->addConstData(
            ieData->getName(),
            descriptor,
            ieBlobContent(ieBlob, descriptor.type()));

        bindData(vpuData, ieData);
    }
}

}  // namespace vpu
