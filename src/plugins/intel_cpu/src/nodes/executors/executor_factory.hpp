// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "executor.hpp"
#include "memory_format_filter.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/printers.hpp"
#include "nodes/executors/variable_executor.hpp"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

template <typename Attrs>
class ExecutorFactory {
public:
    using ExecutorImplementationRef = std::reference_wrapper<const ExecutorImplementation<Attrs>>;

    ExecutorFactory(Attrs attrs,
                    ExecutorContext::CPtr context,
                    const MemoryDescArgs& descriptors,
                    const MemoryFormatFilter& memoryFormatFilter = {},
                    const std::string& implementationPriority = {})
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)),
          m_suitableImplementations(filter(m_attrs, descriptors, memoryFormatFilter, implementationPriority)) {
        OPENVINO_ASSERT(!m_suitableImplementations.empty(), "No suitable implementations found");
    }

    /**
     * @brief Retrieves the proper memory descriptors based on the provided memory descriptors.
     *
     * Examines the given executor configuration and determines the appropriate
     * memory descriptors to be used.
     *
     * @param descriptors memory descriptors.
     * @return MemoryDescArgs The list of proper memory descriptors based on the configuration.
     * @todo Create proper memory descriptors for all the implementations
     *       to fully enable graph's layout propagation functionality
     */
    [[nodiscard]] std::vector<MemoryDescArgs> getProperMemoryDescriptors(const MemoryDescArgs& descriptors) const {
        DEBUG_LOG("Preconfiguring memory descriptors");

        executor::Config<Attrs> config{descriptors, m_attrs};

        auto getProperMemoryDescArgs = [](const ExecutorImplementationRef& impl,
                                          const executor::Config<Attrs>& config) {
            if (auto optimalConfig = impl.get().createOptimalConfig(config)) {
                return optimalConfig->descs;
            }

            return config.descs;
        };

        std::vector<MemoryDescArgs> memoryDescArgs;
        memoryDescArgs.reserve(m_suitableImplementations.size());
        for (const auto& impl : m_suitableImplementations) {
            memoryDescArgs.emplace_back(getProperMemoryDescArgs(impl, config));
        }

        return memoryDescArgs;
    }

    /**
     * @brief Creates an Executor instance based on the provided memory arguments.
     *
     * Depending on the number of available implementations, returns:
     * - VariableExecutor, if the number of implementations is two or more
     * - Simple Executor, if there is only one available implementation
     *
     * @param memory memory arguments.
     *
     * @return A shared pointer to the created Executor.
     */
    ExecutorPtr make(const MemoryArgs& memory) {
        auto config = createConfig(memory, m_attrs);
        const bool configMatched = std::all_of(m_suitableImplementations.begin(),
                                               m_suitableImplementations.end(),
                                               [&config](const ExecutorImplementationRef& impl) {
                                                   return !impl.get().createOptimalConfig(config).has_value();
                                               });
        OPENVINO_ASSERT(
            configMatched,
            "Failed to create executor since the provided memory arguments do not match the expected configuration");

        // only single executor is available
        if (m_suitableImplementations.size() == 1) {
            const auto& theOnlyImplementation = m_suitableImplementations.front().get();
            return theOnlyImplementation.create(m_attrs, memory, m_context);
        }

        return std::make_shared<VariableExecutor<Attrs>>(memory, m_attrs, m_context, m_suitableImplementations);
    }

private:
    /**
     * @brief Filters and retrieves suitable implementations based on the provided executor configuration.
     *
     * @param attrs The attributes used for filtering implementations.
     * @param descs The memory descriptor arguments.
     * @param implementationPriority Optional. The name of the implementation to prioritize.
     *        If specified, only the implementation with this name will be considered.
     *
     * @note If an implementation is shape agnostic, no further implementations with lower
     *       priority are considered.
     */
    static std::vector<ExecutorImplementationRef> filter(const Attrs& attrs,
                                                         const MemoryDescArgs& descs,
                                                         const MemoryFormatFilter& memoryFormatFilter = {},
                                                         const std::string& implementationPriority = {}) {
        const auto& implementations = getImplementations<Attrs>();
        std::vector<ExecutorImplementationRef> suitableImplementations;
        const executor::Config<Attrs> config{descs, attrs};

        for (const auto& implementation : implementations) {
            DEBUG_LOG("Processing implementation: ", implementation.name());
            if (!implementationPriority.empty() && implementation.name() != implementationPriority) {
                DEBUG_LOG("Implementation: ",
                          implementation.name(),
                          " does not match priority: ",
                          implementationPriority);
                continue;
            }

            if (!implementation.supports(config, memoryFormatFilter)) {
                DEBUG_LOG("Implementation is not supported: ", implementation.name());
                continue;
            }

            suitableImplementations.push_back(std::ref(implementation));

            // implementation is supported and it is shape agnostic, there is no way
            // an implementation with a lower priority will be chosen
            if (implementation.shapeAgnostic()) {
                DEBUG_LOG("Implementation is shape agnostic: ",
                          implementation.name(),
                          ". Stop processing implementations");
                break;
            }
        }

        return suitableImplementations;
    }

    Attrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::vector<ExecutorImplementationRef> m_suitableImplementations;
};

template <typename Attrs>
using ExecutorFactoryPtr = std::shared_ptr<ExecutorFactory<Attrs>>;

template <typename Attrs>
using ExecutorFactoryCPtr = std::shared_ptr<const ExecutorFactory<Attrs>>;

}  // namespace ov::intel_cpu
