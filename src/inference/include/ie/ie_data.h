// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file defines the main Data representation node.
 *
 * @file ie_data.h
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_api.h"
#include "ie_common.h"
#include "ie_layouts.h"
#include "ie_precision.hpp"
#include "ngraph/partial_shape.hpp"

namespace InferenceEngine {

/**
 * @brief This class represents the main Data representation node.
 *
 * The NN graphs are di-graphs consisting of data nodes and layer nodes.
 */
class INFERENCE_ENGINE_API_CLASS(Data) {
    class Impl;

public:
    /**
     * @brief An empty constructor (dimensionless)
     *
     * @param name Name of the data node
     * @param _precision Precision of the data
     * @param layout Data layout
     */
    Data(const std::string& name, Precision _precision, Layout layout = NCHW);

    /**
     * @deprecated Use OpenVINO 2.0 API for dynamic shapes support
     * @brief A constructor with partial shape
     *
     * @param name Name of the data node
     * @param _precision Precision of the data
     * @param shape Partial shape of the data
     * @param layout Data layout
     */
    INFERENCE_ENGINE_DEPRECATED("Use OpenVINO 2.0 API for dynamic shapes support")
    Data(const std::string& name, Precision _precision, const ngraph::PartialShape& shape, Layout layout = BLOCKED);

    /**
     * @brief A constructor with tensor descriptor
     *
     * @param name Name of the data node
     * @param desc Tensor descriptor
     */
    Data(const std::string& name, const TensorDesc& desc);

    /**
     * @brief A copy constructor
     *
     * @param data A data object to copy from
     */
    Data(const Data& data);

    /**
     * @brief An assignment operator
     *
     * @param data A data object to copy from
     * @return An assigned object
     */
    Data& operator=(const Data& data);

    /**
     * @brief Checks if the current node is resolved
     *
     * @return true if resolved, false otherwise.
     */
    bool isInitialized() const;

    /**
     * @brief Sets the data dimensions.
     *
     * After the current node is marked as resolved.
     *
     * @param a_dims Tensor dimensions to set
     */
    void setDims(const SizeVector& a_dims);

    /**
     * @brief Sets the layout value for this Data instance
     *
     * @param layout Layout value to set
     */
    void setLayout(Layout layout);

    /**
     * @brief changes dims and layout at same time
     *
     * @param dims new dimensions
     * @param layout new layout
     */
    void reshape(const SizeVector& dims, Layout layout);

    /**
     * @deprecated Use InferenceEngine::Data::reshape(const SizeVector&, Layout)
     * @brief changes dims and layout at same time
     *
     * @param dims new dimensions
     * @param layout new layout
     */
    INFERENCE_ENGINE_DEPRECATED("Use InferenceEngine::Data::reshape(const SizeVector&, Layout)")
    void reshape(const std::initializer_list<size_t>& dims, Layout layout);

    /**
     * @deprecated Use OpenVINO 2.0 API for dynamic shapes support
     * @brief changes dims and layout at same time
     *
     * @param dims new dimensions
     * @param layout new layout
     */
    INFERENCE_ENGINE_DEPRECATED("Use OpenVINO 2.0 API for dynamic shapes support")
    void reshape(const ngraph::PartialShape& dims, Layout layout);

    /**
     * @brief Gets the layout value for this Data instance
     * @return Layout
     */
    Layout getLayout() const;

    /**
     * @brief Gets Tensor descriptor reference
     *
     * @return reference to TensorDesc
     */
    const TensorDesc& getTensorDesc() const;

    /**
     * @brief Gets a precision type of this Data instance
     *
     * @return Precision type
     */
    const Precision& getPrecision() const;

    /**
     * @brief Sets a precision type of this Data instance
     *
     * @param precision Precision of the data
     */
    void setPrecision(const Precision& precision);

    /**
     * @return data dimensions
     */
    const SizeVector& getDims() const;

    /**
     * @return name of the data object
     */
    const std::string& getName() const;

    /**
     * @brief Sets a name the Data object
     *
     * @param newName Name of the data node
     */

    void setName(const std::string& newName);

    /**
     * @return convenient arbitrary user data holder
     */
    const UserValue& getUserObject() const;

    /**
     * @deprecated Use OpenVINO 2.0 API for dynamic shapes support
     * @brief Checks if current data has dynamic shapes
     * @return true if data has dynamic shapes
     */
    INFERENCE_ENGINE_DEPRECATED("Use OpenVINO 2.0 API for dynamic shapes support")
    bool isDynamic() const;

    /**
     * @deprecated Use OpenVINO 2.0 API for dynamic shapes support
     * @brief Returns partial shapes
     * @return shapes which can have dynamic dimensions
     */
    INFERENCE_ENGINE_DEPRECATED("Use OpenVINO 2.0 API for dynamic shapes support")
    const ngraph::PartialShape& getPartialShape() const;

    /**
     * @private
     * @brief Don't touch this field. An implementation details for Data object.
     */
    std::shared_ptr<Impl> _impl;

private:
    /**
     * @brief A unique name that identifies this data node
     */
    std::string name;

    /**
     * @brief A user utility place holder
     */
    UserValue userObject;

    /**
     * @brief A tensor descriptor
     */
    mutable TensorDesc tensorDesc;
};
}  // namespace InferenceEngine
