// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp/ie_infer_request.hpp"
#include "ie_common.h"
#include "ie_input_info.hpp"
#include "openvino/core/node_output.hpp"
#include "so_ptr.hpp"

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START
class IExecutableNetworkInternal;
class IVariableStateInternal;

/**
 * @interface IInferRequestInternal
 * @brief An internal API of synchronous inference request to be implemented by plugin,
 * which is used in InferRequestBase forwarding mechanism
 * @ingroup ie_dev_api_infer_request_api
 */
class INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CLASS(IInferRequestInternal)
    : public std::enable_shared_from_this<IInferRequestInternal> {
public:
    /**
     * @brief A shared pointer to a IInferRequestInternal interface
     */
    using Ptr = std::shared_ptr<IInferRequestInternal>;

    IInferRequestInternal() = default;

    /**
     * @brief      Constructs a new instance.
     * @param[in]  networkInputs   The network inputs info
     * @param[in]  networkOutputs  The network outputs data
     */
    IInferRequestInternal(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs);

    /**
     * @brief      Constructs a new instance.
     * @param[in]  inputs   The network inputs
     * @param[in]  outputs  The network outputs
     */
    IInferRequestInternal(const std::vector<std::shared_ptr<const ov::Node>>& networkInputs,
                          const std::vector<std::shared_ptr<const ov::Node>>& networkOutputs);

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of InferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void Infer();

    /**
     * @brief The minimal infer function to be implemented by plugins. It infers specified input(s) in synchronous mode
     * @note
     *  * This method is used in IInferRequestInternal::Infer, which calls the common code first and after uses this
     * plugin dependent implementation.
     *  * Blocks all method of InferRequest while request is ongoing (running or waiting in queue)
     */
    virtual void InferImpl();

    /**
     * @brief Cancel current inference request execution
     */
    virtual void Cancel();

    /**
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer.
     *  Note: not all plugins may provide meaningful data
     *  @return - a map of layer names to profiling information for that layer.
     */
    virtual std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const;

    /**
     * @brief Set input/output data to infer
     * @note Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual void SetBlob(const std::string& name, const Blob::Ptr& data);

    /**
     * @brief Get input/output data to infer
     * @note Memory allocation doesn't happen
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input
     * precision and size.
     */
    virtual Blob::Ptr GetBlob(const std::string& name);

    /**
     * @brief Queries memory states.
     * @return Returns memory states
     */
    virtual std::vector<std::shared_ptr<IVariableStateInternal>> QueryState();

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     * @note The method returns immediately. Inference starts also immediately.
     */
    virtual void StartAsync();

    /**
     * @brief The minimal asynchronous inference function to be implemented by plugins.
     * It starts inference of specified input(s) in asynchronous mode
     * @note
     *  * The methos is used in AsyncInferRequestInternal::StartAsync which performs common steps first and
     *  calls plugin dependent implementation of this method after.
     *  * It returns immediately. Inference starts also immediately.
     */
    virtual void StartAsyncImpl();

    /**
     * @brief Waits for the result to become available. Blocks until specified millis_timeout has elapsed or the result
     * becomes available, whichever comes first.
     * @param millis_timeout - maximum duration in milliseconds to block for
     * @note There are special cases when millis_timeout is equal some value of WaitMode enum:
     * * STATUS_ONLY - immediately returns request status (InferRequest::StatusCode). It doesn't block or interrupt
     * current thread.
     * * RESULT_READY - waits until inference result becomes available
     * @return A status code
     */
    virtual StatusCode Wait(int64_t millis_timeout);

    /**
     * @brief Alias for callback type
     */
    using Callback = std::function<void(std::exception_ptr)>;

    /**
     * @brief Set callback function which will be called on success or failure of asynchronous request
     * @param callback - function to be called with the following description:
     */
    virtual void SetCallback(Callback callback);

    /**
     * @brief      Check that @p blob is valid. Throws an exception if it's not.
     *
     * @param[in]  blob     The blob to check
     * @param[in]  name     The name of input or output depending of if the @p blob is input or output
     * @param[in]  isInput  Indicates if @p is input
     * @param[in]  refDims  The reference dims, empty if not specified
     */
    void checkBlob(const Blob::Ptr& blob, const std::string& name, bool isInput, const SizeVector& refDims = {}) const;

    /**
     * @brief      Check that all of the blobs is valid. Throws an exception if it's not.
     */
    virtual void checkBlobs();

    /**
     * @brief      Sets the pointer to executable network internal.
     * @note       Needed to correctly handle ownership between objects.
     * @param[in]  exeNetwork  The executable network
     */
    void setPointerToExecutableNetworkInternal(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork);

    /**
     * @brief      Returns the pointer to executable network internal.
     * @returns    The executable network
     */
    std::shared_ptr<IExecutableNetworkInternal> getPointerToExecutableNetworkInternal() const;

    /**
     * @brief      Sets the pointer to so when needed.
     * @note       Needed to correctly handle ownership between objects.
     * @param[in]  so The library so
     */
    void setPointerToSo(const std::shared_ptr<void>& so);

    /**
     * @brief      Returns the pointer to so.
     * @returns    The library
     */
    std::shared_ptr<void> getPointerToSo() const;
    /**
     * @brief   Gets the pointer to userData.
     * @return  Pointer to user data
     */
    INFERENCE_ENGINE_DEPRECATED("The method will be removed")
    void* GetUserData() noexcept;

    /**
     * @brief       Sets the pointer to userData.
     * @param[in]   Pointer to user data
     */
    INFERENCE_ENGINE_DEPRECATED("The method will be removed")
    void SetUserData(void* userData) noexcept;

    const std::vector<std::shared_ptr<const ov::Node>>& GetInputs() const;
    const std::vector<std::shared_ptr<const ov::Node>>& GetOutputs() const;

    /**
     * @brief Sets inputs/outputs from ov::Model
     */
    virtual void setModelInputsOutputs(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs);

protected:
    /**
     * @brief Destroys the object.
     */
    virtual ~IInferRequestInternal();

    /**
     * @brief Checks and executes input data pre-processing if needed.
     * @param inputs Inputs blobs to perform preprocessing on
     * @param serial Whether to use multiple threads to execute the step
     */
    void execDataPreprocessing(InferenceEngine::BlobMap& preprocessedBlobs, bool serial = false);

    /**
     * @brief Helper function to find input or output blob by name
     * @param name A name of input or output blob.
     * @param foundInput A pointer to input information if found.
     * @param foundOutput A pointer to output DataPtr if found.
     * @return `True` - if loaded network has input with provided name,
     *         `false` - if loaded network has output with provided name
     * @throws [not_found] exception if there is no input and output layers with given name
     */
    bool findInputAndOutputBlobByName(const std::string& name, InputInfo::Ptr& foundInput, DataPtr& foundOutput) const;

    /**
     * @brief Helper function to find input node by legacy blob name
     * @param name A legacy name of input blob
     * @return shared pointer to OpenVINO input node
     */
    std::shared_ptr<const ov::Node> findInputByNodeName(const std::string& name) const;
    /**
     * @brief Helper function to find output node by legacy blob name
     * @param name A legacy name of output blob
     * @return shared pointer to OpenVINO output node
     */
    std::shared_ptr<const ov::Node> findOutputByNodeName(const std::string& name) const;

    /**
     * @brief Checks whether pre-processing step is required for a given input
     * @param info InputInfo corresponding to input blob
     * @param userBlob Input Blob object corresponding to input info
     * @param deviceBlob Blob object in plugin's desired format
     * @return `True` if pre-processing is required, `false` otherwise
     */
    bool preProcessingRequired(const InputInfo::Ptr& info,
                               const Blob::Ptr& userBlob,
                               const Blob::Ptr& deviceBlob = nullptr);

    void addInputPreProcessingFor(const std::string& name, const Blob::Ptr& from, const Blob::Ptr& to);

    InferenceEngine::InputsDataMap _networkInputs;    //!< Holds information about network inputs info
    InferenceEngine::OutputsDataMap _networkOutputs;  //!< Holds information about network outputs data
    InferenceEngine::BlobMap _inputs;                 //!< A map of user passed blobs for network inputs
    InferenceEngine::BlobMap _deviceInputs;           //!< A map of actual network inputs, in plugin specific format
    InferenceEngine::BlobMap _outputs;                //!< A map of user passed blobs for network outputs
    std::vector<std::shared_ptr<const ov::Node>> _parameters;  //!< A vector of function inputs
    std::vector<std::shared_ptr<const ov::Node>> _results;     //!< A vector of function outputs

    /**
     * @brief A shared pointer to IInferRequestInternal
     * @note Needed to correctly handle ownership between objects.
     */
    std::shared_ptr<IExecutableNetworkInternal> _exeNetwork;
    /**
     * @brief A shared pointer to loaded library
     * @note Needed to correctly handle ownership between objects.
     */
    std::shared_ptr<void> _so;
    Callback _callback;  //!< A callback

private:
    void* _userData = nullptr;
};

/**
 * @brief SoPtr to IInferRequestInternal.
 */
using SoIInferRequestInternal = ov::SoPtr<IInferRequestInternal>;

IE_SUPPRESS_DEPRECATED_END

}  // namespace InferenceEngine
