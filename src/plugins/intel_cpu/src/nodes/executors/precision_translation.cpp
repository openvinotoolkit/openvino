// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_translation.hpp"

#include <cassert>
#include <cstdlib>

#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "precision_matcher.hpp"

namespace ov::intel_cpu {

TypeOfArg getTypeConfiguration(const MemoryDescArgs& descriptors,
                               const TypeMapping& mapping,
                               const MappingNotation& notation) {
    assert(!mapping.empty() && !mapping.front().mask().empty());

    InOutTypes types(mapping.front().mask().size());
    // gather types from memory descriptors
    for (const auto& [argId, desc] : descriptors) {
        if (notation.count(argId)) {
            types.at(notation.at(argId)) = desc->getPrecision();
        }
    }
    // match types against the mapping
    TypeOfArg typeConfig;
    for (const auto& entry : mapping) {
        if (!entry.enabled()) {
            continue;
        }

        const auto& pattern = entry.mask();
        if (!match(pattern, types)) {
            continue;
        }

        for (const auto& [argId, desc] : descriptors) {
            if (!notation.count(argId)) {
                continue;
            }

            const size_t id = notation.at(argId);
            types[id] = descriptors.at(argId)->getPrecision();
            typeConfig[argId] = entry.translate(types, id);
        }

        return typeConfig;
    }

    OPENVINO_THROW("Failed to create a type configuration for the provided memory descriptors");
}

}  // namespace ov::intel_cpu
