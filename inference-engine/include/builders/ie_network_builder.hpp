// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_builder.hpp>
#include <ie_icnn_network.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_network.hpp>
#include <ie_context.hpp>
#include <ie_common.h>
#include <ie_blob.h>
#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief This class implements a builder for IE Network
 */
class INFERENCE_ENGINE_API_CLASS(Network): public INetwork {
public:
    /**
     * @brief A shared pointer to the Network builder
     */
    using Ptr = std::shared_ptr<Network>;
    /**
     * @brief An iterator for Network builder definition
     */
    using iterator = details::INetworkIterator<Network, Layer>;
    /**
     * @brief Begin network iterator
     * @return Network iterator
     */
    iterator begin();
    /**
     * @brief Begin network iterator
     * @return const INetwork iterator
     */
    const_iterator begin() const noexcept override;

    /**
     * @brief End network iterator
     * @return Network iterator
     */
    iterator end();
    /**
     * @brief End network iterator
     * @return const INetwork iterator
     */
    const_iterator end() const noexcept override;

    /**
     * @brief Returns a number of layers in the network.
     * @return Layers count
     */
    size_t size() const noexcept override;

    /**
     * @brief The constructor creates a builder based on ICNNNetwork
     *
     * @param network constant reference to ICNNNetwork object
     */
    explicit Network(const ICNNNetwork& network);
    /**
     * @brief The constructor creates a empty builder with network name
     *
     * @param name Network name
     */
    explicit Network(const std::string& name);
    /**
     * @brief The constructor creates a builder based on INetwork
     *
     * @param network constant reference to INetwork object
     */
    explicit Network(const INetwork& network);

    /**
     * @brief The constructor creates a builder based on ICNNNetwork with custom Context
     *
     * @param network constant reference to ICNNNetwork object
     */
    Network(const Context& ieContext, const ICNNNetwork& network);
    /**
     * @brief The constructor creates a empty builder with network name and custom Context
     *
     * @param name Network name
     */
    Network(const Context& ieContext, const std::string& name);
    /**
     * @brief The constructor creates a builder based on INetwork with custom Context
     *
     * @param network constant reference to INetwork object
     */
    Network(const Context& ieContext, const INetwork& network);

    /**
     * @brief Adds new layer and connects it with previous layers
     *
     * @param inputs Vector with PortInfo objects from previous layers
     * @param layer Layer builder for new layer
     *
     * @return Id of new builder for the current network
     */
    idx_t addLayer(const std::vector<PortInfo>& inputs, const Layer& layer);
    /**
     * @brief Adds new layer
     *
     * @param layer Layer builder for new layer
     *
     * @return Id of new builder for the current network
     */
    idx_t addLayer(const Layer& layer);
    /**
     * @brief Removes a layer by ID
     *
     * @param layerId Layer ID
     */
    void removeLayer(idx_t layerId);

    /**
     * @brief Connects two layers
     *
     * @param input PortInfo object from previous layer
     * @param output PortInfo object from next layer
     */
    void connect(const PortInfo& input, const PortInfo& output);
    /**
     * @brief Removes connection from the network
     *
     * @param connection Connection
     */
    void disconnect(const Connection& connection);

    /**
     * @brief Returns vector of layer builders
     *
     * @return Vector of layer builders
     */
    std::vector<Layer::Ptr>& getLayers();
    /**
     * @brief Returns constant vector of layer builders
     *
     * @return constant vector of layer builders
     */
    const std::vector<Layer::Ptr>& getLayers() const;

    /**
     * @brief Returns a constant smart pointer to a Layer interface.
     * If the layer is missing, returns nullptr.
     * @param id Id of the Layer
     * @return Layer interface smart pointer
     */
    const ILayer::CPtr getLayer(idx_t id) const noexcept override;
    Layer::Ptr getLayer(idx_t layerId);

    /**
     * @brief Returns a constant vector of input layers.
     * @return Vector of input layers
     */
    const std::vector<ILayer::CPtr> getInputs() const noexcept override;
    /**
     * @brief Returns a vector of input layers.
     * @return Vector of input layers
     */
    std::vector<Layer::Ptr> getInputs();

    /**
     * @brief Returns a constant vector of output layers.
     * @return Vector of output layers
     */
    const std::vector<ILayer::CPtr> getOutputs() const noexcept override;
    /**
     * @brief Returns a vector of input layers.
     * @return Vector of input layers
     */
    std::vector<Layer::Ptr> getOutputs();

    /**
     * @brief Returns a constant vector of connections for specific layer.
     * If the layer is missing, returns empty vector.
     * @param layerId layer index
     * @return Vector of connections
     */
    const std::vector<Connection> getLayerConnections(idx_t layerId) const noexcept override;

    /**
     * @brief Returns a constant vector of all connections.
     * @return Vector of connections
     */
    const std::vector<Connection>& getConnections() const;

    /**
     * @brief Returns a network name.
     * @return Network name
     */
    const std::string& getName() const noexcept override;

    /**
     * @brief Returns a network context
     * @return const reference to Context
     */
    const Context& getContext() const noexcept override;
    /**
     * @brief Returns a network context
     * @return reference to Context
     */
    Context& getContext() noexcept;

    /**
     * @brief Builds and validate network
     *
     * @return const shared pointer to INetwork
     */
    const INetwork::CPtr build();

    /**
     * @brief Validates network
     *
    */
    void validate();

    /**
     * @brief The operator builds network
     *
     * @return const shared pointer to INetwork
     */
    explicit operator const INetwork::CPtr();

private:
    std::map<std::string, Parameter> parameters;
};

/**
 * @brief This function converts INetwork to ICNNNetwork
 *
 * @param network constant shared pointer to INetwork object
 * @return constant shared pointer to ICNNNetwork
 */
INFERENCE_ENGINE_API_CPP(const std::shared_ptr<ICNNNetwork>) convertToICNNNetwork(const INetwork::CPtr& network);

}  // namespace Builder

}  // namespace InferenceEngine
