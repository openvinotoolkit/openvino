// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

namespace GNAPluginNS {
namespace request {
class ModelWrapperFactory;

class ModelWrapper {
public:
    class ConstructionPassKey {
    public:
        ConstructionPassKey(const ConstructionPassKey&) = default;
        ~ConstructionPassKey() = default;

    private:
        ConstructionPassKey() = default;
        friend class ModelWrapperFactory;
    };

    ModelWrapper(ConstructionPassKey);
    ~ModelWrapper();

    ModelWrapper(const ModelWrapper&) = delete;
    ModelWrapper(ModelWrapper&&) = delete;
    ModelWrapper& operator=(const ModelWrapper&) = delete;
    ModelWrapper& operator=(ModelWrapper&&) = delete;

    Gna2Model& object();
    const Gna2Model& object() const;

private:
    Gna2Model object_;
};

}  // namespace request
}  // namespace GNAPluginNS
