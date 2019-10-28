// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_builder.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {

/**
 * @brief Neural network builder API
 */
namespace Builder {

/**
 * @brief This class defines the basic functional for layer builders
 */
class INFERENCE_ENGINE_API_CLASS(LayerDecorator) {
public:
    /**
     * @brief The constructor creates layer builders with layer type and layer name
     * @param type Layer type
     * @param name Layer name
     */
    LayerDecorator(const std::string& type, const std::string& name);
    /**
     * @brief The constructor creates layer builders from reference to generic layer builder
     * @param layer pointer to generic layer builder
     */
    explicit LayerDecorator(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates layer builders from reference to generic layer builder
     * @param layer constant pointer to generic layer builder
     */
    explicit LayerDecorator(const Layer::CPtr& layer);
    /**
     * @brief The copy constructor
     * @param rval Source builder
     */
    LayerDecorator(const LayerDecorator& rval);

    /**
     * @brief Copy operator for LayerDecorator
     * @param rval
     * @return Layer builder
     */
    LayerDecorator& operator=(const LayerDecorator& rval);

    /**
     * @brief Virtual destructor
     */
    virtual ~LayerDecorator() = default;

    /**
     * @brief The operator creates generic builder
     * @return Generic builder
     */
    virtual operator Layer() const;

    /**
     * @brief The operator creates generic builder
     * @return Pointer to generic builder
     */
    virtual operator Layer::Ptr();

    /**
     * @brief The operator creates generic builder
     * @return Constant pointer to generic builder
     */
    virtual operator Layer::CPtr() const;

    /**
     * @brief Returns layer type
     * @return Layer type
     */
    const std::string& getType() const;
    /**
     * @brief Returns layer name
     * @return Layer name
     */
    const std::string& getName() const;

protected:
    Layer::Ptr& getLayer();
    const Layer::CPtr getLayer() const;
    void checkType(const std::string& type) const;

    Layer::CPtr cLayer;

private:
    Layer::Ptr layer;
};

}  // namespace Builder

}  // namespace InferenceEngine
