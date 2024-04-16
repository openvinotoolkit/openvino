// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_translation.hpp"

#include <iterator>

#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "precision_matcher.hpp"

namespace ov {
namespace intel_cpu {

InOutTypes getTypeConfiguration(const MemoryDescArgs& descriptors, const TypeMapping& mapping, const MappingNotation& notation) {
    InOutTypes types;
    std::transform(notation.begin(), notation.end(), std::back_inserter(types), [&descriptors](int id) {
        return descriptors.at(id)->getPrecision();
    });

    for (const auto& entry : mapping) {
        if (!entry.enabled())
            continue;

        const auto& pattern = entry.mask();
        if (!match(pattern, types))
            continue;

        return entry.translate(types);
    }

    OPENVINO_THROW("Failed to create a type configuration for the provided memory descriptors");
}

}  // namespace intel_cpu
}  // namespace ov
