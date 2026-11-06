// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mark_grouped_matmul_rows_per_token.hpp"

#include <memory>

#include "intel_gpu/op/moe_expert_sort.hpp"
#include "intel_gpu/op/moe_router_fused.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;
using namespace ov::op;
using ov::pass::operator|;

MarkGroupedMatMulRowsPerToken::MarkGroupedMatMulRowsPerToken() {
    const auto topk_over_last_axis = ov::pass::pattern::op::Predicate(
        [](const Output<Node>& out) {
            const auto topk = ov::as_type<ov::op::util::TopKBase>(out.get_node());
            return topk->get_input_partial_shape(0).rank() == 2 && topk->get_axis() == 1;
        },
        "topk_over_last_axis");
    auto ids_shape = rank_equals(2) && has_static_dim(1);
    auto topk_m = wrap_type_strict_index<v1::TopK, v3::TopK, v11::TopK>({any_input(), any_input()},
                                                                         ids_shape && topk_over_last_axis);
    auto router_m = wrap_type_strict_index<ov::intel_gpu::op::MoERouterFused>({}, ids_shape);
    auto ids_m = topk_m->output(1) | router_m->output(1);
    auto sort_m = wrap_type_strict_index<ov::intel_gpu::op::MoEExpertSort>({optional<v0::Convert>({ids_m})});
    auto offsets_m = optional<v0::Convert>({sort_m->output(1)});
    using GroupedMatMulCompressed = ov::op::internal::GroupedMatMulCompressed;
    auto gmm_m = wrap_type<v17::GroupedMatMul>({any_input(), any_input(), offsets_m}) |
                 wrap_type<GroupedMatMulCompressed>({any_input(), any_input(), offsets_m, any_input()}) |
                 wrap_type<GroupedMatMulCompressed>({any_input(), any_input(), offsets_m, any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto gmm = m.get_match_root();
        if (transformation_callback(gmm)) {
            return false;
        }
        const auto sort = m.get_pattern_map().at(sort_m);
        gmm->get_rt_info()[grouped_matmul_rows_per_token_key] = sort->get_input_partial_shape(0)[1].get_length();
        return false;
    };

    auto m = std::make_shared<Matcher>(gmm_m, "MarkGroupedMatMulRowsPerToken");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
