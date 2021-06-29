// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/utils/ie_helpers.hpp>
#include <vpu/model/data_contents/ie_blob_content.hpp>

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
        const auto& match = env.config.compileConfig().ioStrides.find(name);
        if (match == env.config.compileConfig().ioStrides.end()) {
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
        const auto& netInputName = inputInfo.first;
        const auto paramNodeIter = std::find_if(_ieParsedNetwork.networkParameters.begin(), _ieParsedNetwork.networkParameters.end(), [&netInputName](const NodePtr& node) {
            return node->get_friendly_name() == netInputName;
        });
        IE_ASSERT(paramNodeIter != _ieParsedNetwork.networkParameters.end());
        const auto& paramNode = *paramNodeIter;
        IE_ASSERT(netInput != nullptr);

        const auto ieData = netInput->getInputData();
        IE_ASSERT(ieData != nullptr);

        env.log->trace("Network input : %s", ieData->getName());
        VPU_LOGGER_SECTION(env.log);

        const auto vpuDesc = DataDesc{ieData->getTensorDesc()};
        const auto vpuData = model->addInputData(ieData->getName(), vpuDesc);

        parseIOStrides(netInputName, vpuData);

        bindData(vpuData, paramNode->output(0), paramNode);
    }

    //
    // Parse network outputs
    //

    env.log->trace("Parse network outputs");

    for (const auto& outputInfo : _ieParsedNetwork.networkOutputs) {
        const auto& ieData = outputInfo.second;
        const auto& netOutputName = outputInfo.first;
        IE_ASSERT(ieData != nullptr);

        env.log->trace("Network output : %s", ieData->getName());
        VPU_LOGGER_SECTION(env.log);
        const auto& resultNodeIter = std::find_if(_ieParsedNetwork.networkResults.begin(), _ieParsedNetwork.networkResults.end(), [&netOutputName](const NodePtr& node) {
            return (node->get_input_node_shared_ptr(0)->get_friendly_name() == netOutputName);
        });
        IE_ASSERT(resultNodeIter != _ieParsedNetwork.networkResults.end());
        const auto& resultNode = *resultNodeIter;

        const auto vpuDesc = DataDesc{ieData->getTensorDesc()};
        const auto vpuData = model->addOutputData(ieData->getName(), vpuDesc);

        parseIOStrides(outputInfo.first, vpuData);

        bindData(vpuData, resultNode->get_input_source_output(0), resultNode->get_input_node_shared_ptr(0));

        if (_unbatchedOutputs.count(ieData) > 0) {
            env.log->trace("The output %s is unbatched", vpuData);
            vpuData->attrs().set<bool>("unbatched", true);
        }
    }

    //
    // Parse network constant
    //

    env.log->trace("Parse network constants");

    for (const auto& constNode : _ieParsedNetwork.networkConstants) {
        // const auto& ieData = constInfo.first;
        auto constant = ngraph::as_type_ptr<ngraph::opset4::Constant>(constNode);
        IE_ASSERT(constant != nullptr);

        env.log->trace("Network constant : %s", constant->get_friendly_name());
        VPU_LOGGER_SECTION(env.log);

        // TODO: move method to utility
        // rework

        auto blob = shareWeights(constNode);
        auto descriptor = DataDesc{constant->get_output_tensor(0)};
        if (descriptor.type() == DataType::FP32) {
            descriptor.setType(DataType::FP16);
        }

        const auto vpuData = model->addConstData(
            constant->get_friendly_name(),
            descriptor,
            ieBlobContent(blob, descriptor.type()));

        bindData(vpuData, constNode->output(0), constNode);
    }
}

}  // namespace vpu
