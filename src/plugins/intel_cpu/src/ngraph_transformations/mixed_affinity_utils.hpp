// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <openvino/core/node.hpp>

namespace ov {
namespace intel_cpu {
namespace mixed_affinity {
constexpr ov::label_t batch_label = 2007;

struct Subgraph {
    std::vector<ov::Input<ov::Node>> starts;
    std::vector<ov::Output<ov::Node>> ends;
};

class Properties {
public:
    Properties() : opt_bs(0), n_splits(0) {}
    Properties(const size_t opt_bs, const size_t n_splits) : opt_bs(opt_bs), n_splits(n_splits) {}
    bool is_set() const;
    bool operator<(const Properties& other) const;
    bool operator==(const Properties& other) const;
    bool operator!=(const Properties& other) const;

    size_t opt_bs;
    size_t n_splits;
};

size_t get_batch_idx(const ov::PartialShape& shape);
}  // namespace mixed_affinity
}  // namespace intel_cpu
}  // namespace ov

namespace std {
template <>
struct hash<ov::intel_cpu::mixed_affinity::Properties> {
    size_t operator()(const ov::intel_cpu::mixed_affinity::Properties& other) const;
};

}  // namespace std