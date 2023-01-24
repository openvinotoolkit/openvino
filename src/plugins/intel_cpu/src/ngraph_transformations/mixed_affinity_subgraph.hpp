// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <set>
#include <openvino/core/node.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
struct Subgraph {
    std::set<ov::Input<ov::Node>> starts;
    std::set<ov::Output<ov::Node>> ends;
};

class Characteristics {
public:
    Characteristics(const size_t opt_bs, const size_t n_splits) : opt_bs(opt_bs), n_splits(n_splits) {}
    bool operator<(const Characteristics& other) const { return opt_bs < other.opt_bs || n_splits < other.n_splits; }
    bool operator==(const Characteristics& other) const { return opt_bs == other.opt_bs && n_splits == other.n_splits; }

    size_t opt_bs;
    size_t n_splits;
};
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov

namespace std {
template <>
struct hash<ov::intel_cpu::mixed_affinity::Characteristics> {
    size_t operator()(const ov::intel_cpu::mixed_affinity::Characteristics& other) const {
        return std::hash<size_t>()(other.opt_bs) ^ (std::hash<size_t>()(other.n_splits) << 10);
    }
};

}  // namespace std