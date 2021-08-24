// Copyright (C) 2018-2021 Intel Corporation
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

using ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const std::shared_ptr<ngraph::Node>& op)>;

struct ExtensionsHolder {
    std::map<std::string, ext_factory> list;
};

class MKLDNNExtensions : public IExtension {
public:
    MKLDNNExtensions();

    virtual StatusCode
    getPrimitiveTypes(char**& types, unsigned int& size, ResponseDesc* resp) noexcept {
        collectTypes(types, size);
        return OK;
    }

    virtual StatusCode
    getFactoryFor(ILayerImplFactory*& factory, const std::shared_ptr<ngraph::Node>& op, ResponseDesc* resp) noexcept {
        using namespace MKLDNNPlugin;
        factory = layersFactory.createNodeIfRegistered(MKLDNNPlugin, op->get_type_name(), op);
        if (!factory) {
            std::string errorMsg = std::string("Factory for ") + op->get_type_name() + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return NOT_FOUND;
        }
        return OK;
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

    using LayersFactory = openvino::cc::Factory<
                                std::string,
                                InferenceEngine::ILayerImplFactory*(const std::shared_ptr<ngraph::Node>& op)>;

    LayersFactory layersFactory;

private:
    void collectTypes(char**& types, unsigned int& size) const {
        types = new char *[layersFactory.size()];
        unsigned count = 0;
        layersFactory.foreach([&](std::pair<std::string, LayersFactory::builder_t> const &builder) {
            types[count] = new char[builder.first.size() + 1];
            std::copy(builder.first.begin(), builder.first.end(), types[count]);
            types[count][builder.first.size() ] = '\0';
        });
        size = count;
    }
};

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
