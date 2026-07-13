// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_router_scale.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu {

FuseMoERouterScale::FuseMoERouterScale() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    auto routing_m = any_input();

    auto per_expert_const_m = wrap_type<ov::op::v0::Constant>(rank_equals(1));
    auto topk_idx_m = any_input();
    auto axis_m = wrap_const();
    auto gather_m = wrap_type<ov::op::v8::Gather>({per_expert_const_m, topk_idx_m, axis_m});

    auto scalar_scale_const_m = wrap_const();
    auto multiply_m = wrap_type<ov::op::v1::Multiply>({routing_m, gather_m | scalar_scale_const_m});

    auto w2_scale_m = wrap_const();
    auto moe_compressed_m = wrap_type<ov::op::internal::MOECompressed>({any_input(),
                                                                        multiply_m,
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        any_input(),
                                                                        w2_scale_m,
                                                                        any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pm = m.get_pattern_value_map();
        const auto& w2_scale = pm.at(w2_scale_m);
        auto apply_scale = [&](ov::Output<ov::Node> scale, bool need_unsqueeze) {
            if (scale.get_element_type() != w2_scale.get_element_type()) {
                scale = std::make_shared<ov::op::v0::Convert>(scale, w2_scale.get_element_type());
            }
            if (need_unsqueeze) {
                const auto w2_scale_shape = w2_scale.get_shape();
                const size_t ndim = w2_scale_shape.size();
                std::vector<int32_t> axes(ndim - 1);
                for (size_t i = 0; i < ndim - 1; ++i) {
                    axes[i] = static_cast<int32_t>(i + 1);
                }
                auto axes_const = ov::op::v0::Constant::create(ov::element::i32, {axes.size()}, axes);
                scale = std::make_shared<ov::op::v0::Unsqueeze>(scale, axes_const);
            }
            return std::make_shared<ov::op::v1::Multiply>(w2_scale, scale);
        };

        ov::Output<ov::Node> new_w2_scale;
        if (pm.count(scalar_scale_const_m)) {
            const auto& constant = pm.at(scalar_scale_const_m);
            if (ov::shape_size(constant.get_shape()) != 1) {
                return false;
            }
            new_w2_scale = apply_scale(constant, false);
        } else if (pm.count(per_expert_const_m)) {
            const auto w2_scale_shape = w2_scale.get_shape();
            const auto per_expert_const_shape = pm.at(per_expert_const_m).get_shape();
            if (w2_scale_shape.empty() || per_expert_const_shape.size() != 1 || per_expert_const_shape[0] != w2_scale_shape[0]) {
                return false;
            }
            new_w2_scale = apply_scale(pm.at(per_expert_const_m), true);
        } else {
            return false;
        }
        auto folded = ov::util::get_constant_from_source(new_w2_scale);
        if (!folded) {
            return false;
        }
        ov::copy_runtime_info(w2_scale.get_node_shared_ptr(), folded);

        auto moe_compressed = pm.at(moe_compressed_m).get_node_shared_ptr();
        auto new_inputs = moe_compressed->input_values();
        new_inputs[1] = pm.at(routing_m);
        new_inputs[10] = folded;

        auto new_moe = moe_compressed->clone_with_new_inputs(new_inputs);
        new_moe->set_friendly_name(moe_compressed->get_friendly_name());
        ov::copy_runtime_info(moe_compressed, new_moe);
        ov::replace_node(moe_compressed, new_moe);
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "FuseMoERouterScale");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
