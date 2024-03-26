// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "ov_optional.hpp"

namespace ov {
namespace intel_cpu {

// @todo Consider alternative of using template arguments instead of std::functions
template <typename Attrs>
class ExecutorImplementation {
public:
    using SupportsPredicate = std::function<bool(const executor::Config<Attrs>&)>;
    using RequiresFallbackPredicate = std::function<ov::optional<executor::Config<Attrs>>(const executor::Config<Attrs>&)>;
    using AcceptsShapePredicate = std::function<bool(const MemoryArgs& memory)>;
    using CreateFunction = std::function<ExecutorPtr(const Attrs& attrs,
                                                     const PostOps& postOps,
                                                     const MemoryArgs& memory,
                                                     const ExecutorContext::CPtr context)>;

    ExecutorImplementation(
        const char* name,
        const ExecutorType type,
        const OperationType operationType,
        const ShapeTolerance shapeRelation,
        SupportsPredicate supports,
        RequiresFallbackPredicate requiresFallback,
        AcceptsShapePredicate acceptsShape,
        CreateFunction create)
        : m_name(name),
          m_type(type),
          m_operationType(operationType),
          m_shapeRelation(shapeRelation),
          m_supports(std::move(supports)),
          m_requiresFallback(std::move(requiresFallback)),
          m_acceptsShape(std::move(acceptsShape)),
          m_create(std::move(create)) {}

    bool supports(const executor::Config<Attrs>& config) const {
        if (m_supports) {
            return m_supports(config);
        }

        return false;
    }

    ov::optional<executor::Config<Attrs>> requiresFallback(const executor::Config<Attrs>& config) const {
        if (m_requiresFallback) {
            return m_requiresFallback(config);
        }

        return {};
    }

    bool acceptsShapes(const MemoryArgs& memory) const {
        if (m_acceptsShape) {
            return m_acceptsShape(memory);
        }

        return false;
    }

    ExecutorPtr create(const Attrs& attrs,
                       const PostOps& postOps,
                       const MemoryArgs& memory,
                       const ExecutorContext::CPtr context) const {
        DEBUG_LOG("Creating executor using implementation: ", m_name);

        if (m_create)
            return m_create(attrs, postOps, memory, context);
        return nullptr;
    }

    bool shapeAgnostic() const {
        return m_shapeRelation == ShapeTolerance::Agnostic;
    }

    const char* name() const {
        return m_name;
    }

    const ExecutorType type() const {
        return m_type;
    }

    const OperationType operationType() const {
        return m_operationType;
    }

private:
    const char* m_name;
    ExecutorType m_type;
    OperationType m_operationType;
    ShapeTolerance m_shapeRelation;
    SupportsPredicate m_supports;
    RequiresFallbackPredicate m_requiresFallback;
    AcceptsShapePredicate m_acceptsShape;
    CreateFunction m_create;
};

template <typename Attrs>
using ExecutorImplementationPtr = std::shared_ptr<ExecutorImplementation<Attrs>>;
}  // namespace intel_cpu
}  // namespace ov
