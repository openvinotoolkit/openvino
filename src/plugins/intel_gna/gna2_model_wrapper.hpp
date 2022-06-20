// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gna2-model-api.h>

namespace GNAPluginNS {
class Gna2ModelWrapperFactory;

class Gna2ModelWrapper {
public:
    class ConstructionPassKey {
    public:
        ConstructionPassKey(const ConstructionPassKey&) = default;
        ~ConstructionPassKey() = default;

    private:
        ConstructionPassKey() = default;
        friend class Gna2ModelWrapperFactory;
    };

    Gna2ModelWrapper(ConstructionPassKey);
    ~Gna2ModelWrapper();

    Gna2ModelWrapper(const Gna2ModelWrapper&) = delete;
    Gna2ModelWrapper(Gna2ModelWrapper&&) = delete;
    Gna2ModelWrapper& operator=(const Gna2ModelWrapper&) = delete;
    Gna2ModelWrapper& operator=(Gna2ModelWrapper&&) = delete;

    Gna2Model& object();
    const Gna2Model& object() const;

private:
    Gna2Model object_;
};

}  // namespace GNAPluginNS
