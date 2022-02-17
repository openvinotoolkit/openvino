// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_selective_build.h>

#include <ie_iextension.h>

#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <ngraph/node.hpp>

namespace InferenceEngine {

class ILayerImplFactory {
public:
    /**
     * @brief A shared pointer to the ILayerImplFactory interface
     */
    using Ptr = std::shared_ptr<ILayerImplFactory>;

    using ImplCreator = std::function<ILayerImpl*()>;

    /**
     * @brief Destructor
     */
    virtual ~ILayerImplFactory() = default;

    /**
     * @brief Gets all possible implementations for the given cnn Layer
     *
     * @param impls the vector with implementations which is ordered by priority
     * @param resp response descriptor
     * @return status code
     */
    virtual StatusCode getImplementations(std::vector<ILayerImpl::Ptr>& impls, ResponseDesc* resp) noexcept = 0;
};

namespace Extensions {
namespace Cpu {

// TODO: remove this
class MKLDNNExtensions : public IExtension {
public:
    MKLDNNExtensions();

    virtual StatusCode
    getFactoryFor(ILayerImplFactory*& factory, const std::shared_ptr<ngraph::Node>& op, ResponseDesc* resp) noexcept {
        return NOT_FOUND;
    }

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static Version ExtensionDescription = {
            { 2, 1 },    // extension API version
            "2.1",
            "ie-cpu-ext"  // extension description message
        };

        versionInfo = &ExtensionDescription;
    }

    void Unload() noexcept override {}
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
