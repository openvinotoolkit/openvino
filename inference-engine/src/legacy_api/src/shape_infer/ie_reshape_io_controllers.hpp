// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ie_reshape_launcher.hpp"
#include "shape_infer/built-in/ie_built_in_holder.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

struct ShapeDesc {
    std::string dataName;
    SizeVector dims;
};

class DefaultChecker {
public:
    using Ptr = std::shared_ptr<DefaultChecker>;

    virtual void run(const std::vector<DataPtr>& inData, const std::string& layerName);

    virtual ~DefaultChecker() = default;
};

class EmptyChecker : public DefaultChecker {
public:
    void run(const std::vector<DataPtr>& inData, const std::string& layerName) override {};
};

class InputController {
public:
    InputController(const std::vector<DataPtr>& dataVec, const std::string& layerName,
                    const DefaultChecker::Ptr& checker = std::make_shared<DefaultChecker>());

    virtual ~InputController() = default;

    /**
     * @brief Set shape for current reshape launcher by corresponding Data name.
     * @param shape - shape to be set
     * @param dataName - Data's name
     */
    virtual void setShapeByName(const SizeVector& shape, const std::string& dataName);

    /**
     * @brief Return calculated shape for name.
     */
    virtual SizeVector getShapeByName(const std::string& dataName);

    /**
     * @brief Set shape for current reshape launcher by corresponding index.
     * @param shape - shape to be set
     * @param index - shape's index
     */
    virtual void setShapeByIndex(const SizeVector& shape, size_t index);

    /**
     * @brief Returns shapes that are supposed to be set by reshape algorithm.
     * @note Shapes are in topological order.
     * @param check - indicator whether check for correspondence of input data and shapes is required
     * @return shapes
     */
    virtual std::vector<SizeVector> getShapes(bool check);

    /**
     * @brief Returns shapes from IR. If Controller was initialized irShapesOnInit=false, it accesses Data object of
     * Layer If not, all shapes from IR are collected on Controller's construction.
     * @note Shapes are in topological order.
     * @return shapes from IR
     */
    virtual std::vector<SizeVector> getIRShapes();

    /**
     * @brief Returns shape from IR by corresponding Data's name
     * @param dataName - name of Data object that holds requested shape
     * @return shape from IR
     */
    virtual SizeVector getIRShapeByName(const std::string& dataName);

    /**
     * @brief Applies calculated shapes to the Data of the Layer
     */
    virtual void applyChanges();

    /**
     * @brief Reset vector of input shapes.
     */
    virtual void reset();

    virtual void checkCorrespondence();

    virtual bool isDataAvailable();

    virtual std::vector<Blob::CPtr> getBlobs(bool check);

    virtual void setBlobByName(const Blob::CPtr& blob, const std::string& name);

private:
    long getPositionByName(const std::string& dataName);

protected:
    std::vector<DataPtr> _dataVec;
    std::vector<SizeVector> _shapes;
    std::vector<SizeVector> _irShapes;
    std::vector<std::string> _dataNames;
    std::string _layerName;
    std::vector<Blob::CPtr> _inferedData;
};

/**
 * @brief Keeps calculated output shapes, distribute (propagate) them to the connected layers, applies output shapes to
 * the Data object
 */
class OutputController : public InputController {
public:
    OutputController(const std::vector<DataPtr>& inData, const std::string& layerName,
                     const DefaultChecker::Ptr& checker = std::make_shared<DefaultChecker>());

    /**
     * @brief Set calculated output shapes as inputs for next layers
     * @param launchers - Map of layer names to reshape launchers for that layer
     */
    virtual void propagateShapes(const std::set<ReshapeLauncher::Ptr>& launchers);

    virtual void setShapes(const std::vector<SizeVector>& shapes);

    virtual void setBlobs(const std::vector<Blob::Ptr>& blobs);

    std::vector<Blob::Ptr> createBlobs();

    void propagateBlobs(const std::set<ReshapeLauncher::Ptr>& set);
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
