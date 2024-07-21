#pragma once

#include <functional>
#include <memory>
#include <set>

#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/GraphRewrite.hpp"

namespace ov {
using matcher_pass_callback = std::function<bool(pass::pattern::Matcher& m)>;
using graph_rewrite_callback = std::function<bool(pass::pattern::Matcher& m)>;
using handler_callback = std::function<bool(const std::shared_ptr<Node>& node)>;
namespace pass {
class OPENVINO_API BackwardGraphRewrite : public GraphRewrite {
public:
    OPENVINO_RTTI("ov::pass::BackwardGraphRewrite");

    BackwardGraphRewrite() = default;

    explicit BackwardGraphRewrite(const std::shared_ptr<MatcherPass>& pass) : GraphRewrite(pass) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
}  // namespace pass
}  // namespace ov
