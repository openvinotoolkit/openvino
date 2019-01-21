// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_builder.hpp>
#include <ie_icnn_network.hpp>
#include <cpp/ie_cnn_network.h>
#include <ie_inetwork.hpp>
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
class INFERENCE_ENGINE_API_CLASS(Network) {
public:
    /**
     * @brief A shared pointer to the Network builder
     */
    using Ptr = std::shared_ptr<Network>;

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
     * @brief Virtual destructor
     */
    virtual ~Network() = default;

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
     * @brief Returns layer builder by ID
     *
     * @param layerId Layer ID
     *
     * @return Layer buider
     */
    Layer& getLayer(idx_t layerId);
    /**
     * @brief Returns constant layer builder by ID
     *
     * @param layerId Layer ID
     *
     * @return constant layer builder
     */
    const Layer& getLayer(idx_t layerId) const;

    /**
     * @brief Returns vector of layer builders
     *
     * @return Vector of layer builders
     */
    std::vector<Layer>& getLayers();
    /**
     * @brief Returns constant vector of layer builders
     *
     * @return constant vector of layer builders
     */
    const std::vector<Layer>& getLayers() const;

    /**
     * @brief Returns all connections for layer
     *
     * @param layerId Layer ID
     *
     * @return Vector of connections for the current layer
     */
    const std::vector<Connection> getLayerConnections(idx_t layerId) const noexcept;

    /**
     * @brief Builds and validate networks
     *
     * @return const shared pointer to INetwork
     */
    const INetwork::Ptr build() const;

    /**
     * @brief The operator builds network
     *
     * @return const shared pointer to INetwork
     */
    explicit operator const INetwork::Ptr() const;

private:
    const Context ctx;
    const size_t version;
    std::string name;
    std::vector<Layer> layers;
    std::vector<Connection> connections;
};

/**
 * @brief This function converts INetwork to ICNNNetwork
 *
 * @param network constant shared pointer to INetwork object
 * @return constant shared pointer to ICNNNetwork
 */
INFERENCE_ENGINE_API_CPP(const std::shared_ptr<ICNNNetwork>) convertToICNNNetwork(const INetwork::Ptr& network);

}  // namespace Builder

}  // namespace InferenceEngine
