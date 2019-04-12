// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for ArgMax layer
 */
class INFERENCE_ENGINE_API_CLASS(DetectionOutputLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit DetectionOutputLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit DetectionOutputLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit DetectionOutputLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    DetectionOutputLayer& setName(const std::string& name);

    /**
     * @brief Returns output port
     * @return Output port
     */
    const Port& getOutputPort() const;
    /**
     * @brief Sets output port
     * @param port Output port
     * @return reference to layer builder
     */
    DetectionOutputLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns input ports
     * @return Vector of input ports
     */
    const std::vector<Port>& getInputPorts() const;
    /**
     * @brief Sets input ports
     * @param ports Vector of input ports
     * @return reference to layer builder
     */
    DetectionOutputLayer& setInputPorts(const std::vector<Port>& ports);
    /**
     * @brief Returns number of classes
     * @return Number of classes
     */
    size_t getNumClasses() const;
    /**
     * @brief Sets number of classes to be predict
     * @param num Number of classes
     * @return reference to layer builder
     */
    DetectionOutputLayer& setNumClasses(size_t num);
    /**
     * @brief Returns background label ID
     * @return Background ID
     */
    int getBackgroudLabelId() const;
    /**
     * @brief Sets background label ID
     * @param labelId Background ID if there is no background class, set it to -1.
     * @return reference to layer builder
     */
    DetectionOutputLayer& setBackgroudLabelId(int labelId);
    /**
     * @brief Returns maximum number of results to be kept on NMS stage
     * @return Top K
     */
    int getTopK() const;
    /**
     * @brief Sets maximum number of results to be kept on NMS stage
     * @param topK Top K
     * @return reference to layer builder
     */
    DetectionOutputLayer& setTopK(int topK);
    /**
     * @brief Returns number of total boxes to be kept per image after NMS step
     * @return Keep top K
     */
    int getKeepTopK() const;
    /**
     * @brief Sets number of total boxes to be kept per image after NMS step
     * @param topK Keep top K
     * @return reference to layer builder
     */
    DetectionOutputLayer& setKeepTopK(int topK);
    /**
     * @brief Returns number of oriented classes
     * @return Number of oriented classes
     */
    int getNumOrientClasses() const;
    /**
     * @brief Sets number of oriented classes
     * @param numClasses Number of classes
     * @return reference to layer builder
     */
    DetectionOutputLayer& setNumOrientClasses(int numClasses);
    /**
     * @brief Returns type of coding method for bounding boxes
     * @return String with code type
     */
    std::string getCodeType() const;
    /**
     * @brief Sets type of coding method for bounding boxes
     * @param type Type
     * @return reference to layer builder
     */
    DetectionOutputLayer& setCodeType(std::string type);
    /**
     * @brief Returns interpolate orientation
     * @return Interpolate orientation
     */
    int getInterpolateOrientation() const;
    /**
     * @brief Sets interpolate orientation
     * @param orient Orientation
     * @return reference to layer builder
     */
    DetectionOutputLayer& setInterpolateOrientation(int orient);
    /**
     * @brief Returns threshold to be used in NMS stage
     * @return Threshold
     */
    float getNMSThreshold() const;
    /**
     * @brief Sets threshold to be used in NMS stage
     * @param threshold NMS threshold
     * @return reference to layer builder
     */
    DetectionOutputLayer& setNMSThreshold(float threshold);
    /**
     * @brief Returns confidence threshold
     * @return Threshold
     */
    float getConfidenceThreshold() const;
    /**
     * @brief Sets confidence threshold
     * @param threshold Threshold
     * @return reference to layer builder
     */
    DetectionOutputLayer& setConfidenceThreshold(float threshold);
    /**
     * @brief Returns share location
     * @return true if bounding boxes are shared among different classes
     */
    bool getShareLocation() const;
    /**
     * @brief Sets share location
     * @param flag true if bounding boxes are shared among different classes
     * @return reference to layer builder
     */
    DetectionOutputLayer& setShareLocation(bool flag);
    /**
     * @brief Returns encoded settings
     * @return true if variance is encoded in target
     */
    bool getVariantEncodedInTarget() const;
    /**
     * @brief Sets encoded settings
     * @param flag true if variance is encoded in target
     * @return reference to layer builder
     */
    DetectionOutputLayer& setVariantEncodedInTarget(bool flag);
};

}  // namespace Builder
}  // namespace InferenceEngine
