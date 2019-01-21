// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/caseless.hpp>
#include <ie_parameter.hpp>
#include <ie_inetwork.hpp>
#include <ie_blob.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace InferenceEngine {
namespace Builder {

class Layer;

/**
 * @brief This structure implements a holder for validators
 */
struct ValidatorsHolder {
    /**
     * @brief Caseless map connects type with validator
     */
    details::caseless_map<std::string, std::function<void(const Layer&)>> validators;
};

/**
 * @brief This class implements a builder for IE Layer
 */
class INFERENCE_ENGINE_API_CLASS(Layer) {
public:
    /**
     * @brief The constructor creates a Layer builder with layer type and layer name
     * @param type Layer type
     * @param name Layer name
     */
    explicit Layer(const std::string& type, const std::string& name = "");
    /**
     * @brief The constructor creates a Layer builder from shared pointer to ILayer
     * @param layer shared pointer to ILayer
     */
    explicit Layer(const ILayer::Ptr& layer);
    /**
     * @brief The constructor creates a Layer builder from shared pointer to constant ILayer
     * @param layer shared pointer to constant ILayer
     */
    explicit Layer(const ILayer::CPtr& layer);
    /**
     * @brief The constructor creates a Layer builder with layer ID and layer builder
     * @param id Layer ID
     * @param layer layer builder
     */
    Layer(idx_t id, const Layer& layer);

    /**
     * @brief Returns layer builder ID
     * @return ID
     */
    idx_t getId() const;

    /**
     * @brief Returns a reference to layer type
     * @return Layer type
     */
    std::string& getType();
    /**
     * @brief Returns a reference to constant layer type
     * @return constant layer type
     */
    const std::string& getType() const;
    /**
     * @brief Sets layer type
     * @param type Layer type
     * @return Reference to Layer builder
     */
    Layer& setType(const std::string& type);

    /**
     * @brief Returns a reference to layer name
     * @return Layer name
     */
    std::string& getName();
    /**
     * @brief Returns a reference to constant layer name
     * @return constant layer name
     */
    const std::string& getName() const;
    /**
     * @brief Sets layer name
     * @param name Layer name
     * @return Reference to Layer builder
     */
    Layer& setName(const std::string& name);

    /**
     * @brief Returns layer subgraph
     * @return shared pointer to INetwork
     */
    INetwork::Ptr& getGraph();
    /**
     * @brief Returns constant layer subgraph
     * @return constant shared pointer to INetwork
     */
    const INetwork::Ptr& getGraph() const;
    /**
     * @brief Sets layer subgraph
     * @param graph constant shared pointer to INetwork
     * @return Reference to Layer builder
     */
    Layer& setGraph(const INetwork::Ptr& graph);

    /**
     * @brief Returns map of parameters
     * @return map of parameters
     */
    std::map<std::string, Parameter>& getParameters();
    /**
     * @brief Returns constant map of parameters
     * @return constant map of parameters
     */
    const std::map<std::string, Parameter>& getParameters() const;
    /**
     * @brief Sets parameters for layer
     * @param params constant map of parameters
     * @return Reference to Layer builder
     */
    Layer& setParameters(const std::map<std::string, Parameter>& params);

    /**
     * @brief Returns map of internal blobs
     * @return map of internal blobs
     */
    std::map<std::string, Blob::CPtr>& getConstantData();
    /**
     * @brief Returns constant map of internal blobs
     * @return constant map of internal blobs
     */
    const std::map<std::string, Blob::CPtr>& getConstantData() const;
    /**
     * @brief Sets constant data for layer
     * @param constData constant map of shared pointers to blobs
     * @return Reference to Layer builder
     */
    Layer& setConstantData(const std::map<std::string, Blob::Ptr>& constData);
    /**
     * @brief Sets constant data for layer
     * @param constData constant map of shared pointers to constant blobs
     * @return Reference to Layer builder
     */
    Layer& setConstantData(const std::map<std::string, Blob::CPtr>& constData);
    /**
     * @brief Adds constant data for layer by name
     * @param name Name of constant data
     * @param data shared pointer to constant blob
     * @return Reference to Layer builder
     */
    Layer& addConstantData(const std::string& name, const Blob::CPtr& data);

    /**
     * @brief Returns vector of input ports
     * @return Vector of input ports
     */
    std::vector<Port>& getInputPorts();
    /**
     * @brief Returns constant vector of input ports
     * @return constant vector of input ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param ports vector of ports
     * @return Reference to Layer builder
     */
    Layer& setInputPorts(const std::vector<Port> &ports);

    /**
     * @brief Returns vector of output ports
     * @return Vector of output ports
     */
    std::vector<Port>& getOutputPorts();
    /**
     * @brief Returns constant vector of output ports
     * @return constant vector of output ports
     */
    const std::vector<Port>& getOutputPorts() const;
    /**
     * @brief Sets output ports
     * @param ports vector of ports
     * @return Reference to Layer builder
     */
    Layer& setOutputPorts(const std::vector<Port> &ports);

    /**
     * @brief Validates the current builder and generates ILayer object
     * @return constant shared pointer to ILayer
     */
    const ILayer::Ptr build() const;

    /**
     * @brief Validates layer builder
     */
    void validate() const;

    /**
     * @brief Registers a new validator for type
     * @param type Layer type
     * @param validator Layer validator
     */
    static void addValidator(const std::string& type, const std::function<void(const Layer&)>& validator);

private:
    idx_t id;
    std::string type;
    std::string name;
    INetwork::Ptr graph;
    std::vector<Port> inPorts;
    std::vector<Port> outPorts;
    std::map<std::string, Parameter> params;
    std::map<std::string, Blob::CPtr> constData;

    static std::shared_ptr<ValidatorsHolder> getValidatorsHolder();
};

/**
 * @brief This class registers layer validators
 */
class ValidatorRegisterBase {
public:
    /**
     * @brief The constructor registers new layer validator
     * @param type Layer type
     * @param validator Layer validator
     */
    explicit ValidatorRegisterBase(const std::string& type, const std::function<void(const Layer&)>& validator) {
        InferenceEngine::Builder::Layer::addValidator(type, validator);
    }
};

#define REG_VALIDATOR_FOR(__type, __validator) \
static InferenceEngine::Builder::ValidatorRegisterBase _reg_##__type(#__type, __validator)

}  // namespace Builder
}  // namespace InferenceEngine
