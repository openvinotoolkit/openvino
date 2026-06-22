// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_batch_dim_resolver.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

namespace {
// True if `out` is a Constant holding a single element equal to `value`.
bool is_scalar_constant_with_value(const Output<Node>& out, int64_t value) {
    auto constant = ov::as_type_ptr<v0::Constant>(out.get_node_shared_ptr());
    if (!constant)
        return false;
    if (shape_size(constant->get_shape()) != 1)
        return false;
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] == value;
}

// True if `out` is a Constant holding a single positive integer.
bool is_scalar_positive_constant(const Output<Node>& out) {
    auto constant = ov::as_type_ptr<v0::Constant>(out.get_node_shared_ptr());
    if (!constant)
        return false;
    if (shape_size(constant->get_shape()) != 1)
        return false;
    const auto values = constant->cast_vector<int64_t>();
    return values.size() == 1 && values[0] > 0;
}
}  // namespace

ReshapeBatchDimResolver::ReshapeBatchDimResolver() {
    // Match `Reshape(data, Concat(...))`: the offending view-shape is always built as a Concat
    // (see get_input_concat_if_list); a fully const-folded shape has no propagation problem.
    const auto concat_pattern = wrap_type<v0::Concat>();
    const auto reshape_pattern = wrap_type<v1::Reshape>({any_input(), concat_pattern});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto reshape = ov::as_type_ptr<v1::Reshape>(pattern_map.at(reshape_pattern).get_node_shared_ptr());
        auto concat = ov::as_type_ptr<v0::Concat>(pattern_map.at(concat_pattern).get_node_shared_ptr());
        if (!reshape || !concat)
            return false;

        // Only the standard frozen-batch shape: special_zero must be false (a 0 already means
        // "copy dim", which this rewrite would conflict with) and the shape concat is on axis 0.
        if (reshape->get_special_zero())
            return false;
        if (concat->get_axis() != 0)
            return false;

        const auto& shape_inputs = concat->input_values();
        // Need a leading batch dim, at least one interior dim, and a trailing channel slot.
        if (shape_inputs.size() < 3)
            return false;

        // Leading element must be a baked positive-int batch constant (e.g. the traced 1).
        if (!is_scalar_positive_constant(shape_inputs.front()))
            return false;

        // The infer slot (-1) must be unique and sit in the LAST (channel) position. This is the
        // window-reverse signature and excludes ordinary reshapes whose -1 is elsewhere/absent.
        const size_t channel_idx = shape_inputs.size() - 1;
        if (!is_scalar_constant_with_value(shape_inputs.back(), -1))
            return false;
        for (size_t i = 0; i + 1 < shape_inputs.size(); ++i) {
            if (is_scalar_constant_with_value(shape_inputs[i], -1))
                return false;  // more than one -1 -> ambiguous, not our pattern
        }

        // At least one interior dimension (between batch and channel) must be dynamic, i.e. a
        // genuine shape expression. A fully-constant shape vector is never a propagation bug.
        bool has_dynamic_interior = false;
        for (size_t i = 1; i < channel_idx; ++i) {
            if (!ov::as_type_ptr<v0::Constant>(shape_inputs[i].get_node_shared_ptr())) {
                has_dynamic_interior = true;
                break;
            }
        }
        if (!has_dynamic_interior)
            return false;

        // The batch must be dynamic for the frozen leading constant to be wrong. After
        // conversion the PyTorch decoder makes parameter dims dynamic, so this holds for the
        // traced model while excluding genuinely static reshapes.
        const auto& data = reshape->input_value(0);
        const auto& data_ps = data.get_partial_shape();
        if (data_ps.rank().is_static() && data_ps[0].is_static())
            return false;

        // Rebuild the shape vector for THIS reshape's own data (the concat may be shared between
        // blocks, so we must not edit it in place):
        //   leading batch  -> -1          (inferred from the real element count)
        //   channel (-1)    -> Gather(ShapeOf(data), last)  (data's last dim, == the channel)
        ov::pass::NodeRegistry rg;
        const auto shape_of = rg.make<v3::ShapeOf>(data, element::i32);
        const auto last_index = v0::Constant::create(element::i32, Shape{1}, {-1});
        const auto gather_axis = v0::Constant::create(element::i32, Shape{}, {0});
        const auto channel_dim = rg.make<v8::Gather>(shape_of, last_index, gather_axis);

        const auto& channel_et = concat->input_value(channel_idx).get_element_type();
        Output<Node> channel_out = channel_dim;
        if (channel_et.is_static() && channel_et != element::i32)
            channel_out = rg.make<v0::Convert>(channel_dim, channel_et);

        const auto& batch_et = shape_inputs.front().get_element_type();
        const auto minus_one =
            rg.make<v0::Constant>(batch_et.is_static() ? batch_et : element::i64, Shape{1}, std::vector<int64_t>{-1});

        OutputVector new_shape_inputs;
        new_shape_inputs.reserve(shape_inputs.size());
        new_shape_inputs.push_back(minus_one);
        for (size_t i = 1; i < channel_idx; ++i)
            new_shape_inputs.push_back(shape_inputs[i]);
        new_shape_inputs.push_back(channel_out);

        const auto new_concat = rg.make<v0::Concat>(new_shape_inputs, 0);
        reshape->input(1).replace_source_output(new_concat);
        copy_runtime_info_and_name(concat, rg.get(), {reshape});
        return true;
    };

    auto m = std::make_shared<Matcher>(reshape_pattern, "ov::frontend::pytorch::pass::ReshapeBatchDimResolver");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
