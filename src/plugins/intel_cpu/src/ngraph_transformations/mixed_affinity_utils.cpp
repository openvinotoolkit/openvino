// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity_utils.hpp"
#include <dimension_tracker.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
bool Characteristics::operator<(const Characteristics& other) const {
    return opt_bs < other.opt_bs || n_splits < other.n_splits;
}

bool Characteristics::operator==(const Characteristics& other) const {
    return opt_bs == other.opt_bs && n_splits == other.n_splits;
}

size_t get_batch_idx(const ov::PartialShape& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (ov::DimensionTracker::get_label(shape[i]) == batch_label)
            return i;
    }
    return 0ul;
}

}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov

namespace std {
size_t hash<ov::intel_cpu::mixed_affinity::Characteristics>::operator()(const  ov::intel_cpu::mixed_affinity::Characteristics& other) const {
    return std::hash<size_t>()(other.opt_bs) ^ (std::hash<size_t>()(other.n_splits) << 10);
}
}  // namespace std
