// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Android-only integration
#if defined(ANDROID) || defined(__ANDROID__)

#include "transformations/snippets/normalize_convert_pre_lower.hpp"

#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/opsets/opset1.hpp>

#include "snippets/op/subgraph.hpp"
#include "snippets/op/convert_truncation.hpp"

namespace ov::intel_cpu::pass {

namespace {
class ReplaceConvertInBody : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CPU::ReplaceConvertInBody");
    ReplaceConvertInBody() {
        auto m_v0 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>();
        auto m_v1 = ov::pass::pattern::wrap_type<ov::opset1::Convert>();

        auto callback = [](ov::pass::pattern::Matcher& m) {
            const auto root = m.get_match_root();
            std::shared_ptr<ov::Node> cv = root;
            ov::element::Type dst;
            if (auto c0 = std::dynamic_pointer_cast<ov::op::v0::Convert>(cv)) {
                dst = c0->get_destination_type();
            } else if (auto c1 = std::dynamic_pointer_cast<ov::opset1::Convert>(cv)) {
                dst = c1->get_destination_type();
            } else {
                return false;
            }
            auto in = cv->input_value(0);
            auto ct = std::make_shared<ov::snippets::op::ConvertTruncation>(in, dst);
            ct->set_friendly_name(cv->get_friendly_name());
            ov::copy_runtime_info(cv, ct);
            ov::replace_node(cv, ct);
            return true;
        };

        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_v0, "CPUReplaceConvertV0"), callback);
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_v1, "CPUReplaceConvertV1"), callback);
    }
};
}  // namespace

NormalizeConvertPreLower::NormalizeConvertPreLower() {
    auto m_subgraph = ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>();
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_subgraph, get_type_info_static().name),
                     [](ov::pass::pattern::Matcher& m) {
                         auto s = std::dynamic_pointer_cast<ov::snippets::op::Subgraph>(m.get_match_root());
                         if (!s)
                             return false;
                         ov::pass::Manager body_mgr("CPU::NormalizeConvertPreLowerBody");
                         body_mgr.set_per_pass_validation(false);
                         body_mgr.register_pass<ReplaceConvertInBody>();
                         body_mgr.run_passes(s->body_ptr());
                         return true;
                     });
}

}  // namespace ov::intel_cpu::pass

#endif  // defined(ANDROID) || defined(__ANDROID__)
