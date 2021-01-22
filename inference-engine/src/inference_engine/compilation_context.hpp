// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    explicit NetworkCompilationContext(CNNNetwork network,
        const std::map<std::string, std::string> & compileOptions = {}) :
            m_compileOptions(compileOptions),
            m_inputsInfo(network.getInputsInfo()),
            m_outputsInfo(network.getOutputsInfo()) {
        try {
            auto & icnnnet = static_cast<ICNNNetwork &>(network);
            auto & ngraphImpl = dynamic_cast<details::CNNNetworkNGraphImpl &>(icnnnet);

            std::stringstream xmlFile, binFile;
            ngraph::pass::Serialize serializer(xmlFile, binFile,
                ngraph::pass::Serialize::Version::IR_V10, ngraphImpl.getExtensions());
            serializer.run_on_function(ngraphImpl.getFunction());

            m_constants = xmlFile.str();
            m_model = binFile.str();
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
                m_runtime_atrributes += op->get_friendly_name() + "#" + affinity->get();
            }

            auto priorities_it = rt.find("PrimitivesPriority");
            if (rt.end() != priorities_it) {
                auto primPriority = std::dynamic_pointer_cast<ngraph::VariantWrapper<std::string>>(priorities_it->second);
                m_runtime_atrributes += op->get_friendly_name() + "#" + primPriority->get();
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
        seed = hash_combine(seed, m_constants);

        for (const auto & kvp : m_compileOptions) {
            seed = hash_combine(seed, kvp.first + kvp.second);
        }

        seed = hash_combine(seed, m_runtime_atrributes);

        for (const auto & input : m_inputsInfo) {
            InputInfo::Ptr info = input.second;
            seed = hash_combine(seed, as_int32_t(info->getPrecision()));
            seed = hash_combine(seed, as_int32_t(info->getLayout()));

            const InferenceEngine::PreProcessInfo& preproc = info->getPreProcess();
            seed = hash_combine(seed, as_int32_t(preproc.getResizeAlgorithm()));
            seed = hash_combine(seed, as_int32_t(preproc.getColorFormat()));
            seed = hash_combine(seed, as_int32_t(preproc.getMeanVariant()));

            if (preproc.getMeanVariant() == MeanVariant::MEAN_VALUE) {
                seed = hash_combine(seed, preproc.getNumberOfChannels());
                for (size_t c = 0; c < preproc.getNumberOfChannels(); ++c) {
                    const PreProcessChannel::Ptr & channelInfo = preproc[c];
                    seed = hash_combine(seed, channelInfo->stdScale);
                    seed = hash_combine(seed, channelInfo->meanValue);
                }
            } else if (preproc.getMeanVariant() == MeanVariant::MEAN_IMAGE) {
                // TODO: think if we need to compute hash for mean image if it exists
            }
        }

        for (const auto & output : m_outputsInfo) {
            DataPtr info = output.second;
            seed = hash_combine(seed, as_int32_t(info->getPrecision()));
            seed = hash_combine(seed, as_int32_t(info->getLayout()));
        }

        return std::to_string(seed);
    }

private:
    template <typename T>
    static int32_t as_int32_t(T v) {
        return static_cast<int32_t>(v);
    }

    template <typename T>
    static std::size_t hash_combine(std::size_t seed, const T & a) {
        std::size_t val = std::hash<T>()(a);

        // Hash combine formula from boost
        return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    bool m_cachingIsAvailable = true;

    // network structure (ngraph::Function description)
    std::string m_constants;
    std::string m_model;

    // compile options
    std::map<std::string, std::string> m_compileOptions;

    // runtime information
    std::string m_runtime_atrributes;
    InferenceEngine::InputsDataMap m_inputsInfo;
    InferenceEngine::OutputsDataMap m_outputsInfo;
};

}  // namespace InferenceEngine
