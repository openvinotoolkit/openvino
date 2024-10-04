// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>

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

    using Axes = std::vector<std::size_t>;
    std::map<PPtr, Axes> closures_to_permute;
    void permute(PPtr orig_param, const Axes& order);

    std::set<PPtr> closures_to_f16;
    void to_f16(PPtr orig_param);

    using Ref = std::reference_wrapper<Context>;
};

class DQMatMulGQi : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQi(Context::Ref ctx);
};

class DQMatMulGQ2i : public ov::pass::MatcherPass {
public:
    explicit DQMatMulGQ2i(Context::Ref ctx);
};

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
