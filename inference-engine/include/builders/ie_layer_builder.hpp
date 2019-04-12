// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/caseless.hpp>
#include <ie_parameter.hpp>
#include <ie_network.hpp>
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
    details::caseless_map<std::string, std::function<void(const std::shared_ptr<const Layer>&, bool)>> validators;
};

/**
 * @brief This class implements a builder for IE Layer
 */
class INFERENCE_ENGINE_API_CLASS(Layer): public ILayer,
        public std::enable_shared_from_this<Layer> {
public:
    /**
     * @brief A shared pointer to the Layer builder
     */
    using Ptr = std::shared_ptr<Layer>;
    /**
     * @brief A shared pointer to the constant Layer builder
     */
    using CPtr = std::shared_ptr<const Layer>;

    /**
     * @brief The constructor creates a Layer builder with layer type and layer name
     * @param type Layer type
     * @param name Layer name
     */
    explicit Layer(const std::string& type, const std::string& name = "");
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
     * @brief Compares the given Layer builder with the current one
     * @param rhs Layer builder to compare with
     * @return true if the given Layer builder is equal to the current one, false - otherwise
     */
    bool operator==(const Layer& rhs) const {
        return params == rhs.params;
    }

    /**
     * @brief Returns layer ID
     * @return Layer ID
     */
    idx_t getId() const noexcept override;

    /**
     * @brief Returns a constant reference to layer name
     * @return Layer name
     */
    const std::string& getName() const noexcept override;
    /**
     * @brief Sets layer name
     * @param name Layer name
     * @return Reference to Layer builder
     */
    Layer& setName(const std::string& name);

    /**
     * @brief Returns a constant reference to layer type
     * @return Layer type
     */
    const std::string& getType() const noexcept override;
    /**
     * @brief Sets layer type
     * @param type Layer type
     * @return Reference to Layer builder
     */
    Layer& setType(const std::string& type);

    /**
     * @brief Returns map of parameters
     * @return map of parameters
     */
    const std::map<std::string, Parameter>& getParameters() const noexcept override;
    /**
     * @brief Returns map of parameters
     * @return map of parameters
     */
    std::map<std::string, Parameter>& getParameters();
    /**
     * @brief Sets parameters for layer
     * @param params constant map of parameters
     * @return Reference to Layer builder
     */
    Layer& setParameters(const std::map<std::string, Parameter>& params);

    /**
     * @brief Returns vector of input ports
     * @return Vector of input ports
     */
    const std::vector<Port>& getInputPorts() const noexcept override;
    /**
     * @brief Returns vector of input ports
     * @return Vector of input ports
     */
    std::vector<Port>& getInputPorts();
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
    const std::vector<Port>& getOutputPorts() const noexcept override;
    /**
     * @brief Returns vector of output ports
     * @return Vector of output ports
     */
    std::vector<Port>& getOutputPorts();
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
    const ILayer::CPtr build() const;

    /**
     * @brief Validates layer builder
     */
    void validate(bool partial = false) const;

    /**
     * @brief Registers a new validator for type
     * @param type Layer type
     * @param validator Layer validator
     */
    static void addValidator(const std::string& type, const std::function<void(const Layer::CPtr&, bool)>& validator);

private:
    idx_t id;
    std::string type;
    std::string name;
    std::vector<Port> inPorts;
    std::vector<Port> outPorts;
    std::map<std::string, Parameter> params;
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
    explicit ValidatorRegisterBase(const std::string& type, const std::function<void(const Layer::CPtr&, bool)>& validator) {
        InferenceEngine::Builder::Layer::addValidator(type, validator);
    }
};

#define REG_VALIDATOR_FOR(__type, __validator) \
static InferenceEngine::Builder::ValidatorRegisterBase _reg_##__type(#__type, __validator)

}  // namespace Builder
}  // namespace InferenceEngine
