// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <functional>
#include <vector>

#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/precision_support.h"

namespace ov {
namespace intel_cpu {

template <size_t bypassId>
struct use {
    ov::element::Type operator()(const std::vector<ov::element::Type>& types,
                                 size_t idx) const {
        assert(bypassId < types.size());
        return types[bypassId];
    }
};

struct bypass {
    ov::element::Type operator()(const std::vector<ov::element::Type>& types,
                                 size_t idx) const {
        return types[idx];
    }
};

template <ov::element::Type_t type>
struct just {
    ov::element::Type operator()(const std::vector<ov::element::Type>& types,
                                 size_t idx) const {
        // ignore everything
        (void)types;
        (void)idx;
        return type;
    }
};

template <>
struct just<TypeMaskAlias::fxx> {
    ov::element::Type operator()(const std::vector<ov::element::Type>& types,
                                 size_t idx) const {
        // ignore everything
        (void)types;
        (void)idx;
        return defaultFloatPrecision();
    }
};

using policy = std::function<ov::element::Type(const std::vector<ov::element::Type>&, size_t idx)>;

struct PortsTranslation {
    template <typename... Policies>
    PortsTranslation(Policies... policies) :
        m_policies{policies...} {}

    std::vector<ov::element::Type> operator()(
        const std::vector<ov::element::Type>& types) const {
        assert(types.size() == m_policies.size());

        std::vector<ov::element::Type> result;
        result.reserve(types.size());
        for (size_t i = 0; i < types.size(); i++) {
            result.emplace_back(m_policies[i](types, i));
        }

        return result;
    }
private:
    std::vector<policy> m_policies;
};

// @todo vectors can be replaced with arrays, since we know the size beforehand
//       pros: should be more efficient and safe
//       cons: more template instances (binary size) of the translation utility functions
using InOutTypes = std::vector<ov::element::Type>;
using TypeTranslationFunction = std::function<InOutTypes(const InOutTypes&)>;
using InOutTypeMask = std::vector<TypeMask>;

class TypeMappingEntry {
public:
    using EnabledPredicate = std::function<bool(void)>;

    TypeMappingEntry(InOutTypeMask mask,
                     TypeTranslationFunction translation,
                     EnabledPredicate enabled = {})
        : m_mask(std::move(mask)),
          m_translation(std::move(translation)),
          m_enabled(std::move(enabled)) {}

    const InOutTypeMask& mask() const {
        return m_mask;
    }

    InOutTypes translate(const InOutTypes& types) const {
        if (m_translation)
            return m_translation(types);
        return {};
    }

    bool enabled() const {
        if (m_enabled)
            return m_enabled();
        return true;
    }

private:
    InOutTypeMask m_mask;
    TypeTranslationFunction m_translation;
    EnabledPredicate m_enabled;
};

using TypeMapping = std::vector<TypeMappingEntry>;
using MappingNotation = std::vector<int>;
using pt = PortsTranslation;

InOutTypes getTypeConfiguration(const MemoryDescArgs& descriptors, const TypeMapping& mapping, const MappingNotation& notation);

}  // namespace intel_cpu
}  // namespace ov
