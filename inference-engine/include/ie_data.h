// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file defines the main Data representation node.
 * @file ie_data.h
 */
#pragma once

#include <map>
#include <memory>
#include <vector>
#include "ie_api.h"
#include "ie_common.h"
#include "details/ie_exception.hpp"
#include "ie_precision.hpp"
#include "ie_layouts.h"
#include <string>

namespace InferenceEngine {
/**
 * @brief This class represents the main Data representation node.
 *
 * The NN graphs are di-graphs consisting of data nodes and layer nodes.
 */
class INFERENCE_ENGINE_API_CLASS(Data) {
public:
    /**
     * @deprecated Use Data::getPrecision
     * @brief A precision type of this Data instance
     */
    INFERENCE_ENGINE_DEPRECATED
    Precision precision;
    /**
     * @deprecated Use Data::getFormat
     * @brief A data layout of this Data instance
     */
    INFERENCE_ENGINE_DEPRECATED
    Layout layout;
    /**
     * @deprecated Use Data::getDims
     * @brief A tensor dimension array (the order is opposite to the order in the IR: w,h,c,n) of this Data instance
     */
    INFERENCE_ENGINE_DEPRECATED
    SizeVector dims;
    /**
     * @deprecated Use Data::getCreatorLayer
     * @brief A pointer to the layer that creates this data element, null for input data elements
     */
    INFERENCE_ENGINE_DEPRECATED
    CNNLayerWeakPtr creatorLayer;
    /**
     * @deprecated Use Data::getName
     * @brief A unique name that identifies this data node
     */
    INFERENCE_ENGINE_DEPRECATED
    std::string name;
    /**
     * @deprecated Use Data::getInputTo
     * @brief A map of layers that use this node as input.
     * It is useful for recursive NN graph traversal.
     */
    INFERENCE_ENGINE_DEPRECATED
    std::map<std::string, CNNLayerPtr> inputTo;
    /**
     * @deprecated Use Data::getUserObject
     * @brief A user utility place holder
     */
    INFERENCE_ENGINE_DEPRECATED
    UserValue userObject;

    /**
     * @brief An empty constructor (dimensionless)
     * @param name Name of the data node
     * @param _precision Precision of the data
     * @param layout Data layout
     */
    Data(const std::string &name, Precision _precision, Layout layout = NCHW);

    /**
     * @brief A full constructor (with dimensions)
     * @param name Name of the data node
     * @param a_dims Data tensor dimensions
     * @param _precision Precision of the data
     * @param layout Data layout
     */
    Data(const std::string &name, const SizeVector &a_dims, Precision _precision, Layout layout = NCHW);
    /**
     * @brief A constructor with tensor descriptor
     * @param name Name of the data node
     * @param desc Tensor descriptor
     */
    Data(const std::string &name, const TensorDesc& desc);

    /**
     * @brief A copy constructor
     * @param data A data
     */
    Data(const Data & data);

    /**
     * @brief A destructor
     */
    ~Data();

    /**
     * @brief An assignment operator
     */
    Data & operator = (const Data &);

    /**
     * @brief Checks if the current node is resolved
     * @return true if resolved, false otherwise.
     */
    bool isInitialized() const;

    /**
     * @brief Sets the data dimensions.
     * After the current node is marked as resolved.
     * @param a_dims Tensor dimensions to set
     */
    void setDims(const SizeVector &a_dims);

    /**
    * @deprecated Use Data::setDims to set batch size.
    * @brief Sets the batch value in the data dimensions.
    * Batch is defined as the last element in the dimensions vector.
    * @param batch_size Batch size to set
    */
    INFERENCE_ENGINE_DEPRECATED
    void setBatchSize(size_t batch_size);

    /**
    * @brief Sets the layout value for this Data instance
    * @param layout Layout value to set
    */
    void setLayout(Layout layout);

    /**
     * @brief changes dims and layout at same time
     * @param dims new dimensions
     * @param layout new layout
     */
    void reshape(const SizeVector &dims, Layout layout);

    /**
    * @brief Gets the layout value for this Data instance
    */
    Layout getLayout() const;

    /**
    * @brief Gets Tensor descriptor reference
      @return reference to TensorDesc
    */
    const TensorDesc& getTensorDesc() const;

    /**
     * @brief Gets a precision type of this Data instance
     * @return Precision type
     */
    const Precision& getPrecision() const;

    /**
     * @brief Sets a precision type of this Data instance
     * @param precision Precision of the data
     */
    void setPrecision(const Precision& precision);

    /**
     * @return data dimensions
     */
    const SizeVector& getDims() const;

    /**
     * @return owner of this data layer, parent layer in di-graph
     */
    CNNLayerWeakPtr& getCreatorLayer();

    /**
     * @return name of the data object
     */
    const std::string& getName() const;


    /**
     * @brief Sets a name the Data object
     * @param newName Name of the data node
     */

    void setName(const std::string& newName);

    /**
     * @brief returns child layers in di-graph
     */
    std::map<std::string, CNNLayerPtr>& getInputTo();

    /**
     * @return convenient arbitrary user data holder
     */
    const UserValue& getUserObject() const;
private:
    mutable TensorDesc tensorDesc;
};
}  // namespace InferenceEngine
