// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <functional>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

// Model optimization patterns. Triggered by the plugin at the very top
namespace patterns {
namespace opt {

class DQMatMulCWi : public ov::pass::MatcherPass {
public:
    DQMatMulCWi();
};

struct Context {
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;

    std::vector<PPtr> closures_to_transpose;

    struct View {
        std::size_t axis;
        std::size_t splits;
        std::size_t idx;
        bool operator< (const View &rhs) const {
            return std::make_tuple(axis, splits, idx) < std::make_tuple(rhs.axis, rhs.splits, rhs.idx);
        }
    };
    std::map< std::pair<PPtr, View>, PPtr > closure_views;
    PPtr view(PPtr orig_param, const View &v);

    using Axes = std::vector<std::size_t>;
    void permute(PPtr orig_param, const Axes &order);
    std::map<PPtr, Axes> closures_to_permute;

    using Ref = std::reference_wrapper<Context>;
};

class DQMatMulGQi : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQi(Context::Ref ctx);
};

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
