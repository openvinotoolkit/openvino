// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

namespace GNAPluginNS {
namespace request {
class ModelWrapperFactory;

/**
 * @class Wrapper to ensure that c struct of gna model will be released in case of destruction.
 */
class ModelWrapper {
public:
    /**
     * @class Construction restriction key to ensure that wrapper will be created only by its factory.
     */
    class ConstructionPassKey {
    public:
        ConstructionPassKey(const ConstructionPassKey&) = default;
        ~ConstructionPassKey() = default;

    private:
        ConstructionPassKey() = default;
        friend class ModelWrapperFactory;
    };

    /**
     * Construct {ModelWrapper} object.
     */
    ModelWrapper(ConstructionPassKey);

    /**
     * Destroy {ModelWrapper} object.
     */
    ~ModelWrapper();

    ModelWrapper(const ModelWrapper&) = delete;
    ModelWrapper(ModelWrapper&&) = delete;
    ModelWrapper& operator=(const ModelWrapper&) = delete;
    ModelWrapper& operator=(ModelWrapper&&) = delete;

    /**
     * @brief Return reference to the wrapped model object
     */
    Gna2Model& object();

    /**
     * @brief Return const reference to the wrapped model object
     */
    const Gna2Model& object() const;

private:
    Gna2Model object_;
};

}  // namespace request
}  // namespace GNAPluginNS
