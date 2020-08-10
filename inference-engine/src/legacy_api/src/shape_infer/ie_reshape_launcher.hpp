// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <legacy/ie_layers.h>
#include "legacy/shape_infer/built-in/ie_built_in_holder.hpp"
#include "shape_infer/const_infer/ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

class InputController;

class OutputController;

class DefaultInitializer {
public:
    using Ptr = std::shared_ptr<DefaultInitializer>;

    IE_SUPPRESS_DEPRECATED_START
    virtual void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl);
    IE_SUPPRESS_DEPRECATED_END

    virtual InputController* createInputController(const CNNLayer* layer);

    virtual OutputController* createOutputController(const CNNLayer* layer);

    virtual ~DefaultInitializer() = default;
};

/**
 * @class ReshapeLauncher
 * @brief Helper class to infer shapes for the given CNNLayer by using specified implementation.
 * Encapsulate input and output shapes, before applying it to the real CNNLayer and Data.
 */
class ReshapeLauncher {
public:
    using Ptr = std::shared_ptr<ReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief constructor
     * @param layer - const pointer to the layer for performing shape inference.
     * It is used to obtain parameters, input/output shapes.
     * @param impl - implementation of shape inference for the given layer
     */
    ReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                    const DefaultInitializer::Ptr& initializer = std::make_shared<DefaultInitializer>());

    virtual void setShapeInferImpl(const IShapeInferImpl::Ptr& impl);

    IE_SUPPRESS_DEPRECATED_END

    virtual ~ReshapeLauncher();

    /**
     * @brief Set input shape for current reshape launcher.
     * @param shape - input shape to be set
     */
    virtual void setShapeByName(const SizeVector& shape, const std::string& dataName);

    virtual void setBlobByName(const Blob::CPtr& blob, const std::string& dataName);

    /**
     * @brief Return calculated shape for data with requested name.
     * @return Result shape
     */
    virtual SizeVector getShapeByName(const std::string& dataName);

    /**
     * @brief Set input shape from IR by Data name. If there's no Data with given name it throws exception
     * @param dataName - name of the corresponding Data.
     */
    virtual void setIRShapeByName(const std::string& dataName);

    /**
     * @brief Calculates output shapes and changed layer params using input shapes that was set
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @param launchers - Map of pairs: layer name and its reshape launcher.
     * @return Status code of the operation. OK if succeeded
     */
    virtual void reshape(const std::set<ReshapeLauncher::Ptr>& launchers);

    virtual void constInfer(const std::set<ReshapeLauncher::Ptr>& launchers);

    /**
     * @brief Apply new input shapes, calculated output shapes and changed layer's params to CNNLayer and Data.
     * @param layer - pointer to the layer for setting changes in layer's params
     */
    virtual void applyChanges(CNNLayer* layer);

    /**
     * @brief Reset all stored to the initial state: input/output shapes and layer's params.
     * @param layer - pointer to the layer for setting changes in layer's params
     */
    virtual void reset();

    virtual std::string getLayerName() const;

    virtual std::string getLayerType() const;

    virtual const CNNLayer* getLayer() const;

protected:
    InputController* _iController = nullptr;
    OutputController* _oController = nullptr;
    const CNNLayer* _layer;

    IE_SUPPRESS_DEPRECATED_START
    IShapeInferImpl::Ptr _reshapeImpl;
    IConstInferImpl::Ptr _inferImpl;
    IE_SUPPRESS_DEPRECATED_END

protected:
    /**
     * @brief Check that all shape infer operations were done with specified layer.
     * @param layer - pointer to the layer to compare with
     */
    void checkLayer(CNNLayer* layer);
};

class FakeInitializer : public DefaultInitializer {
public:
    IE_SUPPRESS_DEPRECATED_START
    void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override;
    IE_SUPPRESS_DEPRECATED_END

    InputController* createInputController(const CNNLayer* layer) override;

    OutputController* createOutputController(const CNNLayer* layer) override;
};

/**
 * @class FakeReshapeLauncher
 * @brief Helper class to infer shapes for layers without registered shape infer functions.
 * Encapsulates input and output shapes, before applying it to the real CNNLayer and Data.
 * If input shape is the same as in IR, it takes output shape from IR as is.
 * It sets batch size to the first output dimension of all outputs if:
 *      1) first dimension of all input layers should be the same (assume this is batch size)
 *      2) calculated input shape of the unsupported layer is different only in a first dimension from original input
 * shape in IR.
 */
class FakeReshapeLauncher : public ReshapeLauncher {
public:
    using Ptr = std::shared_ptr<FakeReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START
    FakeReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl);
    IE_SUPPRESS_DEPRECATED_END

    void reshape(const std::set<ReshapeLauncher::Ptr>& launchers) override;

    void constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) override {}
};

class OutputOnlyInitializer : public DefaultInitializer {
public:
    IE_SUPPRESS_DEPRECATED_START
    void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override;
    IE_SUPPRESS_DEPRECATED_END

    InputController* createInputController(const CNNLayer* layer) override;

    OutputController* createOutputController(const CNNLayer* layer) override;
};

/**
 * @class OutputOnlyReshapeLauncher
 * @brief Helper class to infer shapes for layers without inputs. It creates output controller only, input one is null.
 */
class OutputOnlyReshapeLauncher : public ReshapeLauncher {
public:
    using Ptr = std::shared_ptr<OutputOnlyReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START
    OutputOnlyReshapeLauncher(
        const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
        const OutputOnlyInitializer::Ptr& initializer = std::make_shared<OutputOnlyInitializer>());
    IE_SUPPRESS_DEPRECATED_END

    void setShapeByName(const SizeVector& shape, const std::string& dataName) override;

    void setIRShapeByName(const std::string& dataName) override;

    void applyChanges(CNNLayer* layer) override;

    void reset() override;

    void setBlobByName(const Blob::CPtr& blob, const std::string& dataName) override;

    void constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) override;
};

class InputInitializer : public OutputOnlyInitializer {
public:
    IE_SUPPRESS_DEPRECATED_START
    void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override;
    IE_SUPPRESS_DEPRECATED_END
};

/**
 * @class InputReshapeLauncher
 * @brief Helper class to infer shapes for input layers. Supported layer types: `Input` or `Memory`(as inputs only, if
 * index=1) It takes new given input shape and propagate for connected layers. If shape is not set, it takes shapes from
 * IR.
 */
class InputReshapeLauncher : public OutputOnlyReshapeLauncher {
public:
    using Ptr = std::shared_ptr<InputReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START
    InputReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl,
                         const DefaultInitializer::Ptr& initializer = std::make_shared<InputInitializer>());
    IE_SUPPRESS_DEPRECATED_END

    void reshape(const std::set<ReshapeLauncher::Ptr>& launchers) override;
};

class ConstInitializer : public OutputOnlyInitializer {
public:
    IE_SUPPRESS_DEPRECATED_START
    void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override;
    IE_SUPPRESS_DEPRECATED_END
};

/**
 * @class ConstReshapeLauncher
 * @brief Helper class to infer shapes for layers with Const type.
 * It checks if new given shape is the same as in IR. The launcher fails if not and propagate for connected layers
 * otherwise. If shape is not set, it propagates shapes from IR.
 */
class ConstReshapeLauncher : public OutputOnlyReshapeLauncher {
public:
    using Ptr = std::shared_ptr<InputReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START
    ConstReshapeLauncher(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl);
    IE_SUPPRESS_DEPRECATED_END

    void reshape(const std::set<ReshapeLauncher::Ptr>& launchers) override;
};

class OutMemoryInitializer : public DefaultInitializer {
    IE_SUPPRESS_DEPRECATED_START
    void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override;
    IE_SUPPRESS_DEPRECATED_END

    OutputController* createOutputController(const CNNLayer* layer) override;
};

/**
 * @class OutMemoryReshapeLauncher
 * @brief Helper class to infer shapes for layers with Memory type (as outputs only, if index=0).
 * It sets new input shapes and doesn't call propagation as this layer doesn't have childs.
 */
class OutMemoryReshapeLauncher : public ReshapeLauncher {
public:
    using Ptr = std::shared_ptr<InputReshapeLauncher>;

    IE_SUPPRESS_DEPRECATED_START
    OutMemoryReshapeLauncher(const CNNLayer* layer1, const IShapeInferImpl::Ptr& impl1);
    IE_SUPPRESS_DEPRECATED_END

    void reshape(const std::set<ReshapeLauncher::Ptr>& launchers) override {}

    void applyChanges(CNNLayer* layer) override;

    void reset() override;

    void constInfer(const std::set<ReshapeLauncher::Ptr>& launchers) override {}
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
