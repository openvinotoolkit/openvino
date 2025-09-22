// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "memory_format_filter.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

// @todo Consider alternative of using template arguments instead of std::functions
template <typename Attrs>
class ExecutorImplementation {
public:
    using SupportsExtendedPredicate = std::function<bool(const executor::Config<Attrs>&, const MemoryFormatFilter&)>;
    using SupportsSimplePredicate = std::function<bool(const executor::Config<Attrs>&)>;

    using CreateOptimalConfigPredicate =
        std::function<std::optional<executor::Config<Attrs>>(const executor::Config<Attrs>&)>;
    using AcceptsShapePredicate = std::function<bool(const Attrs& attrs, const MemoryArgs& memory)>;
    using CreateFunction =
        std::function<ExecutorPtr(const Attrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context)>;

    ExecutorImplementation(const char* name,
                           const ExecutorType type,
                           const OperationType operationType,
                           SupportsExtendedPredicate supports,
                           CreateOptimalConfigPredicate createOptimalConfig,
                           AcceptsShapePredicate acceptsShape,
                           CreateFunction create)
        : m_name(name),
          m_type(type),
          m_operationType(operationType),
          m_supports(std::move(supports)),
          m_createOptimalConfig(std::move(createOptimalConfig)),
          m_acceptsShape(std::move(acceptsShape)),
          m_create(std::move(create)) {}

    ExecutorImplementation(const char* name,
                           const ExecutorType type,
                           const OperationType operationType,
                           SupportsSimplePredicate supports,
                           CreateOptimalConfigPredicate createOptimalConfig,
                           AcceptsShapePredicate acceptsShape,
                           CreateFunction create)
        : m_name(name),
          m_type(type),
          m_operationType(operationType),
          m_supports([supports](const executor::Config<Attrs>& config, const MemoryFormatFilter&) {
              return supports(config);
          }),
          m_createOptimalConfig(std::move(createOptimalConfig)),
          m_acceptsShape(std::move(acceptsShape)),
          m_create(std::move(create)) {}

    [[nodiscard]] bool supports(const executor::Config<Attrs>& config,
                                const MemoryFormatFilter& memoryFormatFilter) const {
        if (m_supports) {
            return m_supports(config, memoryFormatFilter);
        }

        return false;
    }

    [[nodiscard]] std::optional<executor::Config<Attrs>> createOptimalConfig(
        const executor::Config<Attrs>& config) const {
        if (m_createOptimalConfig) {
            return m_createOptimalConfig(config);
        }

        return {};
    }

    [[nodiscard]] bool acceptsShapes(const Attrs& attrs, const MemoryArgs& memory) const {
        if (m_acceptsShape) {
            return m_acceptsShape(attrs, memory);
        }

        return true;
    }

    [[nodiscard]] ExecutorPtr create(const Attrs& attrs,
                                     const MemoryArgs& memory,
                                     const ExecutorContext::CPtr context) const {
        DEBUG_LOG("Creating executor using implementation: ", m_name);

        if (m_create) {
            return m_create(attrs, memory, context);
        }
        return nullptr;
    }

    [[nodiscard]] bool shapeAgnostic() const {
        return !static_cast<bool>(m_acceptsShape);
    }

    [[nodiscard]] const char* name() const {
        return m_name;
    }

    [[nodiscard]] ExecutorType type() const {
        return m_type;
    }

    [[nodiscard]] OperationType operationType() const {
        return m_operationType;
    }

private:
    const char* m_name;
    ExecutorType m_type;
    OperationType m_operationType;
    SupportsExtendedPredicate m_supports;
    CreateOptimalConfigPredicate m_createOptimalConfig;
    AcceptsShapePredicate m_acceptsShape;
    CreateFunction m_create;
};

template <typename Attrs>
using ExecutorImplementationPtr = std::shared_ptr<ExecutorImplementation<Attrs>>;
}  // namespace ov::intel_cpu
