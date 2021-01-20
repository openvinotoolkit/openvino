// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cstdint>
#include <map>

#include <transformations/serialize.hpp>
#include <cnn_network_ngraph_impl.hpp>

#include <cpp/ie_cnn_network.h>
#include <details/ie_exception.hpp>

#include <ngraph/variant.hpp>
#include <ngraph/function.hpp>

namespace InferenceEngine {

class NetworkCompilationContext final {
public:
    explicit NetworkCompilationContext(const CNNNetwork & network,
        const std::map<std::string, std::string> & compileOptions = {}) :
            m_compileOptions(compileOptions) {
        try {
            const auto & ngraphImpl = dynamic_cast<const details::CNNNetworkNGraphImpl &>(
                        static_cast<const ICNNNetwork&>(network));
            std::map<std::string, ngraph::OpSet> custom_opsets;
            for (auto extension : ngraphImpl._ie_extensions) {
                auto opset = extension->getOpSets();
                custom_opsets.insert(std::begin(opset), std::end(opset));
            }
            ngraph::pass::Serialize serializer(ngraph::pass::Serialize::Version::IR_V10, custom_opsets);
            serializer.run_on_function(ngraphImpl._ngraph_function);

            m_constants = serializer.getWeights();
            m_model = serializer.getModel();
        } catch (const std::bad_cast &) {
            // IR v7 or older is passed: cannot cast to CNNNetworkNGraphImpl
            m_cachingIsAvailable = false;
            std::cout << "IR v7 is passed; skip import and export" << std::endl;
        } catch (const ngraph::ngraph_error & ex) {
            // failed to serialize the model - caching is not available
            m_cachingIsAvailable = false;
            std::cout << ex.what() << std::endl;
            std::cout << "failed to serialize the model; skip import and export" << std::endl;
        }

        if (!m_cachingIsAvailable)
            return;

        // put "affinity", "PrimitivesPriority" runtime information

        for (const auto & op : network.getFunction()->get_ordered_ops()) {
            ngraph::Node::RTMap rt = op->get_rt_info();

            auto affinity_it = rt.find("affinity");
            if (rt.end() != affinity_it) {
                auto affinity = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(affinity_it->second);
                m_affinities += op->get_friendly_name() + "#" + affinity->get();
            }

            auto priorities_it = rt.find("PrimitivesPriority");
            if (rt.end() != priorities_it) {
                auto primPriority = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(priorities_it->second);
                m_affinities += op->get_friendly_name() + "#" + primPriority->get();
            }
        }

        // information about precisions, layouts


        // information about preprocessing
    }

    bool isCachingAvailable() const {
        return m_cachingIsAvailable;
    }

    std::string computeHash() const {
        IE_ASSERT(isCachingAvailable());

        size_t seed {};
        seed = hash_combine(seed, m_model);

        for (auto & value : m_constants) {
            seed = hash_combine(seed, value);
        }

        for (const auto & kvp : m_compileOptions) {
            seed = hash_combine(seed, kvp.first + kvp.second);
        }

        seed = hash_combine(seed, m_affinities);

        // TODO: more values to hash
        // 1. precisions
        // 2. layouts
        // 3. preprocessing

        return std::to_string(seed);
    }

private:
    template <typename T>
    static std::size_t hash_combine(std::size_t seed, const T & a) {
        std::size_t hash = std::hash<T>()(a);
        return hash ^ (seed << 1);
    }

    bool m_cachingIsAvailable = true;

    // network structure (ngraph::Function description)
    std::string m_constants;
    std::string m_model;;

    // compile options
    std::map<std::string, std::string> m_compileOptions;

    // runtime information
    std::string m_affinities;
};

}  // namespace InferenceEngine
