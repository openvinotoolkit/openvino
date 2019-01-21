// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_builder.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief This class defines the basic functional for layer builders
 */
class INFERENCE_ENGINE_API_CLASS(LayerFragment) {
public:
    /**
     * @brief The constructor creates layer builders with layer type and layer name
     * @param type Layer type
     * @param name Layer name
     */
    LayerFragment(const std::string& type, const std::string& name);
    /**
     * @brief The constructor creates layer builders from reference to generic layer builder
     * @param genLayer Generic layer builder
     */
    explicit LayerFragment(Layer& genLayer);
    /**
     * @brief The copy constructor
     * @param rval Source builder
     */
    explicit LayerFragment(const LayerFragment& rval);

    /**
     * @brief Copy operator for LayerFragment
     * @param rval
     * @return Layer builder
     */
    LayerFragment& operator=(const LayerFragment& rval);

    /**
     * @brief Virtual destructor
     */
    virtual ~LayerFragment() = default;

    /**
     * @brief The operator creates generic builder
     * @return Generic builder
     */
    virtual operator Layer() const;

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
    const std::vector<size_t> uInts2size_t(const std::vector<unsigned int>& vector) const;
    Layer& getLayer() const;

private:
    Layer layer;
    Layer& refLayer;
};

}  // namespace Builder

}  // namespace InferenceEngine
