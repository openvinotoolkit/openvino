// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <ie_iinfer_request.hpp>
#include <ie_primitive_info.hpp>
#include <cpp_interfaces/interface/ie_imemory_state_internal.hpp>

namespace InferenceEngine {

/**
 * @brief minimum API to be implemented by plugin, which is used in ExecutableNetworkBase forwarding mechanism.
 */
class IExecutableNetworkInternal {
public:
    typedef std::shared_ptr<IExecutableNetworkInternal> Ptr;

    virtual ~IExecutableNetworkInternal() = default;


    /**
     * @brief Gets the Executable network output Data node information. The received info is stored in the given Data node.
     * This method need to be called to find output names for using them later during filling of a map
     * of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return out Reference to the ConstOutputsDataMap object
     */
    virtual ConstOutputsDataMap GetOutputsInfo() const = 0;

    /**
     * @brief Gets the Executable network input Data node information. The received info is stored in the given InputsDataMap object.
     * This method need to be called to find out input names for using them later during filling of a map
     * of blobs passed later to InferenceEngine::IInferencePlugin::Infer()
     * @return inputs Reference to ConstInputsDataMap object.
     */
    virtual ConstInputsDataMap GetInputsInfo() const = 0;


    /**
     * @brief Create an inference request object used to infer the network
     *  Note: the returned request will have allocated input and output blobs (that can be changed later)
     * @param req - shared_ptr for the created request
     */
    virtual void CreateInferRequest(IInferRequest::Ptr &req) = 0;

    /**
     * @brief Export the current created executable network so it can be used later in the Import() main API
     * @param modelFileName - path to the location of the exported file
     */
    virtual void Export(const std::string &modelFileName) = 0;

    /**
     * @brief Get the mapping of IR layer names to actual implemented kernels
     * @param deployedTopology - map of PrimitiveInfo objects representing the deployed topology
     */
    virtual void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) = 0;


    virtual std::vector<IMemoryStateInternal::Ptr> QueryState() = 0;
};

}  // namespace InferenceEngine
