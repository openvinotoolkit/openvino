// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "executor.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/graph_emitter.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/printers.hpp"
#include "openvino/core/except.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {
using namespace executor;

template <typename Attrs, typename NodeT>
static ExecutorPtr fallback(const executor::Config<Attrs>& config,
                            const executor::Config<Attrs>& fallbackConfig,
                            const MemoryArgs& memory,
                            const ExecutorContext::CPtr context,
                            const std::string& name) {
    DEBUG_LOG("Falling back to graph executor for ",
              name,
              ". Original config: ",
              config,
              " new config:",
              fallbackConfig);

    GraphEmitter<Attrs> graphEmitter(config.descs, config.attrs, config.postOps, memory, context, name);

    const auto& graphExecutor =
        graphEmitter.createGraph(fallbackConfig.descs, fallbackConfig.attrs, fallbackConfig.postOps, context)
            .ensureAttrsMatch()
            .ensureSrcDescsMatch()
            .ensureDstDescsMatch()
            .ensurePostOpsMatch()
            .emit();
    (void)graphExecutor;

    OPENVINO_THROW("Fallback logic is not implemented yet");  // return graphExecutor;
}

template <typename Attrs, typename NodeT>
class ExecutorFactory {
public:
    using ExecutorImplementationRef = std::reference_wrapper<const ExecutorImplementation<Attrs>>;

    ExecutorFactory(const Attrs& attrs,
                    const PostOps& postOps,
                    const ExecutorContext::CPtr context,
                    const MemoryDescArgs& descriptors,
                    const std::string& implementationPriority = {})
        : m_attrs(attrs),
          m_postOps(postOps),
          m_context(context),
          m_suitableImplementations(filter(m_attrs, m_postOps, descriptors, implementationPriority)),
          m_implementationRequiresFallback(m_suitableImplementations.size(), true),
          m_executors(m_suitableImplementations.size()) {}

    /**
     * @brief Retrieves the proper memory descriptors based on the provided memory descriptors.
     *
     * Examines the given executor configuration and determines the appropriate
     * memory descriptors to be used. Checks for fallback configurations if necessary and
     * returns the corresponding memory descriptors.
     *
     * @param descriptors memory descriptors.
     * @return MemoryDescArgs The proper memory descriptors based on the configuration.
     * @todo Create proper memory descriptors for all the implementations
     *       to fully enable graph's layout propagation functionality
     *
     * @note The main use case is to avoid a fallback during the creation of an executor
     *       by passing proper memory descriptors to the make() method
     */
    MemoryDescArgs getProperMemoryDescriptors(const MemoryDescArgs& descriptors) const {
        DEBUG_LOG("Preconfiguring memory descriptors");

        const auto& impl = m_suitableImplementations.front();
        executor::Config<Attrs> config{descriptors, m_attrs, m_postOps};

        if (auto fallbackConfig = impl.get().requiresFallback(config)) {
            return fallbackConfig->descs;
        }

        return config.descs;
    }

    /**
     * @brief Preconfigures an executor based on the provided memory arguments.
     *
     * Preconfigures an executor by selecting an appropriate implementation based on the provided
     * memory arguments and by creating an executor using the implementation.
     *
     * @param memory The memory parameters used for selecting the appropriate executor implementation.
     *
     * @note The main use case is to offload executor data preparation (i.e. weights packing)
     *       From the make() call
     * @todo Currently supports creating a single executor.
     *       For some nodes it can be worth to preconfigure all the executors.
     */
    void preconfigure(const MemoryArgs& memory) {
        executor::Config<Attrs> config{memoryDescsFromMemory(memory), m_attrs, m_postOps};

        cacheFallbackStatus(config);

        const size_t implId = select(memory, 0);
        const auto& impl = m_suitableImplementations[implId].get();
        DEBUG_LOG("Preconfiguring executor: ", impl.name());

        if (m_implementationRequiresFallback[implId]) {
            if (auto fallbackConfig = impl.requiresFallback(config)) {
                fallback<Attrs, NodeT>(config, *fallbackConfig, memory, m_context, impl.name());
            }
        }

        (void)create(implId, memory, m_context);
    }

    /**
     * @brief Creates an Executor instance based on provided memory arguments.
     *
     * Creates an Executor instance using the provided MemoryArgs, selecting an appropriate implementation
     * based on the characteristics of the memory. It handles fallback scenarios if necessary and updates the executor
     * with the given memory information.
     *
     * @param memory memory arguments.
     *
     * @return A shared pointer to the created Executor.
     *
     * The function follows the steps below:
     * - Selects an implementation based on the provided memory using the select() function.
     * - Retrieves the selected implementation and checks if fallback is required.
     * - If fallback is required, it creates a fallback configuration and returns a fallback executor.
     * - Otherwise creates the executor using the selected implementation.
     * - Updates the executor with the given memory information.
     *
     */
    ExecutorPtr make(MemoryArgs& memory) {
        auto createExec = [this](MemoryArgs& memory, size_t implId) -> ExecutorPtr {
            const auto& impl = m_suitableImplementations[implId].get();
            if (m_implementationRequiresFallback[implId]) {
                executor::Config<Attrs> config{memoryDescsFromMemory(memory), m_attrs, m_postOps};
                if (auto fallbackConfig = impl.requiresFallback(config)) {
                    return fallback<Attrs, NodeT>(config, *fallbackConfig, memory, m_context, impl.name());
                }
            }
            const auto executor = create(implId, memory, m_context);
            if (!executor->update(memory)) {
                return nullptr;
            }
            return executor;
        };

        auto implId = select(memory, 0);
        auto executor = createExec(memory, implId);
        while (!executor) {
            implId = select(memory, ++implId);
            executor = createExec(memory, implId);
        }
        return executor;
    }

private:
    static MemoryDescArgs memoryDescsFromMemory(const MemoryArgs& memory) {
        MemoryDescArgs memoryDescs;
        memoryDescs.reserve(memory.size());

        for (const auto& mem : memory) {
            memoryDescs[mem.first] = mem.second->getDescPtr();
        }

        return memoryDescs;
    }

    /**
     * @brief Caches the fallback status for each suitable implementation.
     */
    void cacheFallbackStatus(const executor::Config<Attrs>& config) {
        std::transform(m_suitableImplementations.begin(),
                       m_suitableImplementations.end(),
                       m_implementationRequiresFallback.begin(),
                       [&config](const ExecutorImplementationRef& impl) {
                           return impl.get().requiresFallback(config);
                       });
    }

    /**
     * @brief Filters and retrieves suitable implementations based on the provided executor configuration.
     *
     * @param attrs The attributes used for filtering implementations.
     * @param postOps The post-operations to be applied.
     * @param descs The memory descriptor arguments.
     * @param implementationPriority Optional. The name of the implementation to prioritize.
     *        If specified, only the implementation with this name will be considered.
     *
     * @note If an implementation is shape agnostic, no further implementations with lower
     *       priority are considered.
     */
    static std::vector<ExecutorImplementationRef> filter(
        const Attrs& attrs,
        const PostOps& postOps,
        const MemoryDescArgs& descs,
        const std::string& implementationPriority = {}) {
        const auto& implementations = getImplementations<Attrs>();
        std::vector<ExecutorImplementationRef> suitableImplementations;
        const executor::Config<Attrs> config{descs, attrs, postOps};

        for (const auto& implementation : implementations) {
            DEBUG_LOG("Processing implementation: ", implementation.name());
            if (!implementationPriority.empty() && implementation.name() != implementationPriority) {
                DEBUG_LOG("Implementation: ",
                          implementation.name(),
                          " does not match priority: ",
                          implementationPriority);
                continue;
            }

            if (!implementation.supports(config)) {
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

    size_t select(const MemoryArgs& memory, const size_t startIdx) const {
        OPENVINO_ASSERT(startIdx < m_suitableImplementations.size(),
            "Failed to find an implementation since start indx: ", startIdx,
            " is out of range of the suitable implementations array: ", m_suitableImplementations.size());
        auto startIt = m_suitableImplementations.begin();
        std::advance(startIt, startIdx);
        const auto selectedImplementation =
            std::find_if(startIt,
                         m_suitableImplementations.end(),
                         [&memory](const ExecutorImplementationRef& implementation) {
                             return implementation.get().shapeAgnostic() || implementation.get().acceptsShapes(memory);
                         });
        OPENVINO_ASSERT(selectedImplementation != m_suitableImplementations.end(), "Failed to select an implemetation");

        return std::distance(m_suitableImplementations.begin(), selectedImplementation);
    }

    ExecutorPtr create(const size_t implId,
                       const MemoryArgs& memory,
                       const ExecutorContext::CPtr context) {
        assert(implId < m_executors.size() && implId < m_suitableImplementations.size());

        if (!m_executors[implId]) {
            const auto& impl = m_suitableImplementations[implId].get();
            m_executors[implId] = impl.create(m_attrs, m_postOps, memory, context);
        }

        return m_executors[implId];
    }

    const Attrs& m_attrs;
    const PostOps& m_postOps;
    const ExecutorContext::CPtr m_context;
    std::vector<ExecutorImplementationRef> m_suitableImplementations;
    // stores fallback status to avoid performing the check for every make() call
    std::vector<bool> m_implementationRequiresFallback;
    // executors cache
    std::vector<ExecutorPtr> m_executors;
};

template <typename Attrs, typename NodeT>
using ExecutorFactoryPtr = std::shared_ptr<ExecutorFactory<Attrs, NodeT>>;

template <typename Attrs, typename NodeT>
using ExecutorFactoryCPtr = std::shared_ptr<const ExecutorFactory<Attrs, NodeT>>;

}  // namespace intel_cpu
}  // namespace ov
