#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_original_precision.hpp"

namespace ov {
namespace runtime {
namespace interpreter {
namespace pass {

// Transformation marks selected nodes with KeepOriginalPrecision attribute.
// With that - ConvertPrecision pass will not convert those nodes' precision.
// It is required for example for Multinomial or RandomUniform as their output
// can be very different (not just by a small margin) with different types (f32 vs. f16 for example).

template <typename... T>
class KeepNodeInOriginalPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KeepNodeInOriginalPrecision", "0");

    KeepNodeInOriginalPrecision() {
        auto root = ov::pass::pattern::wrap_type<T...>();

        matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            ov::set_keep_original_precision_attribute(m.get_match_root());
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "KeepNodeInOriginalPrecision");
        register_matcher(m, callback);
    }
};

}  // namespace pass
}  // namespace interpreter
}  // namespace runtime
}  // namespace ov
