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
        const std::map<std::string, std::string> & compileOptions = {},
        bool print = false) {
        try {
            const auto & ngraphImpl = dynamic_cast<const details::CNNNetworkNGraphImpl &>(
                        static_cast<const ICNNNetwork&>(network));
            std::map<std::string, ngraph::OpSet> custom_opsets;
            for (auto extension : ngraphImpl._ie_extensions) {
                auto opset = extension->getOpSets();
                custom_opsets.insert(std::begin(opset), std::end(opset));
            }
            ngraph::pass::Serialize serializer(
                "", "", ngraph::pass::Serialize::Version::IR_V10, custom_opsets);
            serializer.run_on_function(ngraphImpl._ngraph_function);

            m_constants = std::move(serializer.getWeights());
            m_model = std::move(serializer.getModel());

            std::cout << std::hash<std::string>()(m_model) << " !!!\n";
            if (print)
                std::cout << m_model << std::endl;
        } catch (const std::bad_cast &) {
            // IR v7 or older is passed: cannot cast to CNNNetworkNGraphImpl
            m_cachingIsAvailable = false;
            std::cout << "IR v7 is passed; skip import and export" << std::endl;
        } catch (const ngraph::ngraph_error &) {
            // failed to serialize the model - caching is not available
            m_cachingIsAvailable = false;
            std::cout << "failed to serialize the model; skip import and export" << std::endl;
        }

        // put affinities

        for (const auto & op : network.getFunction()->get_ordered_ops()) {
            ngraph::Node::RTMap rt = op->get_rt_info();
            auto it = rt.find("affinity");
            if (rt.end() != it) {
                auto affinity = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(it->second);
                m_affinities += op->get_friendly_name() + "#" + affinity->get();
            }
        }
    }

    bool isCachingAvailable() const {
        return m_cachingIsAvailable;
    }

    std::string computeHash() const {
        IE_ASSERT(isCachingAvailable());

        size_t seed {};
        seed = hash_combine(seed, m_model);

        // TODO: more values to hash
        // seed = hash_combine(seed, m_constants);
        // seed = hash_combine(seed, m_affinities);

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
    std::vector<uint8_t> m_constants;
    std::string m_model;

    // runtime information
    std::string m_affinities;
};

}  // namespace InferenceEngine
