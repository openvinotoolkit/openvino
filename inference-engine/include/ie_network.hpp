// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for the Inference Engine Network interface
 * @file ie_inetwork.hpp
 */
#pragma once

#include <utility>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <ie_parameter.hpp>
#include <ie_context.hpp>
#include <ie_layouts.h>
#include <ie_blob.h>

namespace InferenceEngine {

/**
 * @brief A type of network objects indexes.
 */
using idx_t = size_t;

/**
 * @brief This class contains a pair from layerId and port index
 */
class PortInfo {
public:
    /**
     * @brief The constructor creates a PortInfo object for port 0
     * @param layerID Layer id
     */
    PortInfo(idx_t layerID): layer(layerID), port(0) {}  // NOLINT

    /**
     * @brief The constructor creates a PortInfo object
     * @param layerID Layer id
     * @param portID Port id
     */
    PortInfo(idx_t layerID, idx_t portID): layer(layerID), port(portID) {}

    /**
     * @brief Get layer id
     * @return Layer id
     */
    idx_t layerId() const {
        return layer;
    }

    /**
     * @brief Get port id
     * @return Port id
     */
    idx_t portId() const {
        return port;
    }

    /**
     * @brief Compares the given PortInfo object with the current one
     * @param portInfo PortInfo object to compare with
     * @return true if the given PortInfo object is equal to the current one, false - otherwise
     */
    bool operator==(const PortInfo& portInfo) const {
        return layer == portInfo.layerId() && port == portInfo.portId();
    }

    /**
     * @brief Checks if the given PortInfo object is not equal to the current one
     * @param portInfo PortInfo object to compare with
     * @return true if the given PortInfo object is not equal to the current one, false - otherwise
     */
    bool operator!=(const PortInfo& portInfo) const {
        return !(*this == portInfo);
    }

private:
    idx_t layer;
    idx_t port;
};

/**
 * @brief This class is the main object to describe the Inference Engine connection.
 */
class Connection {
public:
    /**
     * @brief Constructor of a connection object.
     * @param input pair of the index of input layer and the index of output port
     * @param output pair of the index of output layer and the index of input port
     */
    Connection(const PortInfo& input, const PortInfo& output): input(input), output(output) {}

    /**
     * @brief Compares the given Connection with the current one
     * @param connection Connection to compare with
     * @return true if the given Connection is equal to the current one, false - otherwise
     */
    bool operator==(const Connection& connection) const {
        return input == connection.from() && output == connection.to();
    }

    /**
     * @brief Checks if the given Connection is not equal to the current one
     * @param connection Connection to compare with
     * @return true if the given Connection is not equal to the current one, false - otherwise
     */
    bool operator!=(const Connection& connection) const {
        return !(*this == connection);
    }

    /**
     * Returns a constant reference to a pair of input layer index and output port index.
     * @return pair of the index of input layer and the index of output port
     */
    const PortInfo& from() const {
        return input;
    }

    /**
     * Returns a constant reference to a pair of output layer index and input port index.
     * @return pair of the index of output layer and the index of input port
     */
    const PortInfo& to() const {
        return output;
    }

private:
    PortInfo input;
    PortInfo output;
};

/**
 * This class describes port data
 */
class INFERENCE_ENGINE_API_CLASS(PortData) {
public:
    /**
     * @brief A shared pointer to the PortData object.
     */
    using Ptr = std::shared_ptr<PortData>;

    /**
     * @brief Default constructor
     */
    PortData();

    /**
     * Creates port data with precision and shape
     * @param shape Dimensions
     * @param precision Precision
     */
    PortData(const SizeVector& shape, const Precision& precision);

    /**
     * @brief virtual destructor
     */
    virtual ~PortData() = default;

    /**
     * @brief Returns data
     * @return Blob with data
     */
    const Blob::Ptr& getData() const;

    /**
     * @brief Sets data
     * @param data Blob with data
     */
    void setData(const Blob::Ptr& data);

    /**
     * @brief Returns data parameters
     * @return Map of parameters
     */
    const std::map<std::string, Parameter>& getParameters() const noexcept;

    /**
     * @brief Sets new shapes for data
     * @param shape New shapes
     */
    void setShape(const SizeVector& shape);

private:
    Blob::Ptr data;
    std::map<std::string, Parameter> parameters;

    void createData(const TensorDesc& desc);
};

/**
 * @brief This class is the main object to describe the Inference Engine port.
 */
class INFERENCE_ENGINE_API_CLASS(Port) {
public:
    /**
     * @brief Default constructor of a port object.
     */
    Port();
    /**
     * @brief Constructor of a port object with shapes.
     * @param shapes port shapes
     * @param precision Port precision
     */
    explicit Port(const SizeVector& shapes,
                  const Precision& precision = Precision::UNSPECIFIED);

    /**
     * @brief Copy constructor.
     * @param port object to copy
     */
    Port(const Port& port);

    /**
     * @brief Virtual destructor
     */
    virtual ~Port() = default;

    /**
     * @brief Compares the given Port with the current one
     * @param rhs Port to compare with
     * @return true if the given Port is equal to the current one, false - otherwise
     */
    bool operator== (const Port& rhs) const;

    /**
     * @brief Compares the given Port with the current one
     * @param rhs Port to compare with
     * @return true if the given Port is NOT equal to the current one, false - otherwise
     */
    bool operator!= (const Port& rhs) const;

    /**
     * @brief Returns a constant reference to a vector with shapes.
     * Shapes should be initialized if shape is empty.
     * @return constant reference to shapes
     */
    const SizeVector& shape() const noexcept;

    /**
     * @brief Sets new shapes for current port
     * @param shape New shapes
     */
    void setShape(const SizeVector& shape);

    /**
     * @brief Returns a constant reference to parameters
     * @return Map with parameters
     */
    const std::map<std::string, Parameter>& getParameters() const noexcept;

    /**
     * @brief Sets new parameters for current port
     * @param params New parameters
     */
    void setParameters(const std::map<std::string, Parameter>& params) noexcept;

    /**
     * @brief Sets the new parameter for current port
     * @param name Name of parameter
     * @param param New value
     */
    void setParameter(const std::string& name, const Parameter& param);

    /**
     * @brief Returns port data
     * @return Port data
     */
    const PortData::Ptr& getData() const noexcept;

    /**
     * @brief Sets new port data for current port
     * @param data Port data
     */
    void setData(const PortData::Ptr& data);

private:
    std::map<std::string, Parameter> parameters;
    PortData::Ptr data;
};

class INetwork;
template <class T>
class INetwotkIterator;

/**
 * @brief This class is the main interface to describe the Inference Engine layer.
 * All methods here are constant and do not throw exceptions.
 */
class ILayer {
public:
    /**
     * @brief A shared pointer to the const ILayer object
     */
    using CPtr = std::shared_ptr<const ILayer>;

    /**
     * @brief Virtual destructor for the layer interface
     */
    virtual ~ILayer() = default;

    /**
     * @brief Returns a id of the layer.
     * @return Layer id
     */
    virtual idx_t getId() const noexcept = 0;

    /**
     * @brief Returns a layer name.
     * @return Layer name
     */
    virtual const std::string& getName() const noexcept = 0;

    /**
     * @brief Returns a layer type.
     * @return Layer type
     */
    virtual const std::string& getType() const noexcept = 0;

    /**
     * @brief Returns a constant smart pointer reference to a Parameters interface.
     * @return Parameters interface smart pointer
     */
    virtual const std::map<std::string, Parameter>& getParameters() const noexcept = 0;

    /**
     * @brief Returns a constant reference to a vector with input ports.
     * @return Vector of input ports
     */
    virtual const std::vector<Port>& getInputPorts() const noexcept = 0;

    /**
     * @brief Returns a constant reference to a vector with output ports.
     * @return Vector of output ports
     */
    virtual const std::vector<Port>& getOutputPorts() const noexcept = 0;
};

namespace details {

template<class NT, class LT>
class INetworkIterator;

}  // namespace details

/**
 * @brief This class is the main interface to describe the Inference Engine network.
 *
 * All methods here are constant and do not throw exceptions.
 */
class INetwork {
public:
    /**
     * @brief A shared pointer to the constant INetwork object.
     */
    using CPtr = std::shared_ptr<const INetwork>;
    /**
     * @brief A constant iterator for INetwork definition
     */
    using const_iterator = details::INetworkIterator<const INetwork, const ILayer>;

    /**
     * @brief Virtual destructor for the network interface
     */
    virtual ~INetwork() = default;

    /**
     * @brief Begin network iterator
     * @return const INetwork iterator
     */
    virtual const_iterator begin() const noexcept = 0;

    /**
     * @brief End network iterator
     * @return const INetwork iterator
     */
    virtual const_iterator end() const noexcept = 0;

    /**
     * @brief Returns a number of layers in the network.
     * @return Layers count
     */
    virtual size_t size() const noexcept = 0;

    /**
     * @brief Returns a constant smart pointer to a Layer interface.
     * If the layer is missing, returns nullptr.
     * @param id Id of the Layer
     * @return Layer interface smart pointer
     */
    virtual const ILayer::CPtr getLayer(idx_t id) const noexcept = 0;

    /**
     * @brief Returns a constant vector of input layers.
     * @return Vector of input layers
     */
    virtual const std::vector<ILayer::CPtr> getInputs() const noexcept = 0;

    /**
     * @brief Returns a constant vector of output layers.
     * @return Vector of output layers
     */
    virtual const std::vector<ILayer::CPtr> getOutputs() const noexcept = 0;

    /**
     * @brief Returns a constant vector of connections for specific layer.
     * If the layer is missing, returns empty vector.
     * @param layerId layer index
     * @return Vector of connections
     */
    virtual const std::vector<Connection> getLayerConnections(idx_t layerId) const noexcept = 0;

    /**
     * @brief Returns a network name.
     * @return Network name
     */
    virtual const std::string& getName() const noexcept = 0;

    /**
     * @brief Returns a network context
     * @return const reference to Context
     */
    virtual const Context& getContext() const noexcept = 0;
};

}  // namespace InferenceEngine

#include <details/ie_inetwork_iterator.hpp>
