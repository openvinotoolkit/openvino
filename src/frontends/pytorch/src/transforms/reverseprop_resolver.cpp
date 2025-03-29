// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverseprop_resolver.hpp"

#include <memory>
#include <utility>

#include "helper_ops/gather_assign.hpp"
#include "helper_ops/internal_op.hpp"
#include "helper_ops/slice_assign.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

ReversepropResolver::ReversepropResolver() {
    auto reverse_op = pattern::wrap_type<InternalReverseOperation>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto base_op = m.get_match_root();
        // Apply this transformation only to starting reverse operation
        if (ov::as_type_ptr<InternalReverseOperation>(base_op->get_input_node_shared_ptr(1)))
            return false;

        auto curr_op = base_op;
        std::vector<std::shared_ptr<Node>> rev_ops;
        while (ov::as_type_ptr<InternalReverseOperation>(curr_op)) {
            rev_ops.push_back(curr_op);
            auto target_inputs = curr_op->get_output_target_inputs(0);
            if (target_inputs.size() != 1)
                break;
            curr_op = target_inputs.begin()->get_node()->shared_from_this();
        }
        if (rev_ops.size() < 1)
            return false;

        ov::pass::NodeRegistry rg;
        auto zero = v0::Constant::create(element::i64, Shape{}, {0});
        auto one = v0::Constant::create(element::i64, Shape{}, {1});
        auto neg_one_1d = v0::Constant::create(element::i64, Shape{1}, {-1});
        auto scattering_shape = v0::Constant::create(element::i64, Shape{2}, {-1, 1});

        // Get 1d indices [0..numel) for whole input tensor
        auto start_op = rev_ops.back();
        auto data_to_insert_into = start_op->input_value(0);
        auto input_shape = rg.make<v3::ShapeOf>(data_to_insert_into, element::i64);
        auto numel = rg.make<v1::ReduceProd>(input_shape, zero, false);
        auto full_data_indices_1d = rg.make<v4::Range>(zero, numel, one, element::i64);
        auto full_data_indices = rg.make<v1::Reshape>(full_data_indices_1d, input_shape, false);

        // cut indices in accordance with operations
        Output<Node> data_indices = full_data_indices;
        for (auto it = rev_ops.rbegin(); it != rev_ops.rend(); ++it) {
            curr_op = *it;
            if (ov::as_type_ptr<SliceAssign>(curr_op)) {
                if (curr_op->get_input_size() == 6) {
                    data_indices = rg.make<v8::Slice>(data_indices,
                                                      curr_op->input_value(2),
                                                      curr_op->input_value(3),
                                                      curr_op->input_value(4),
                                                      curr_op->input_value(5));
                } else if (curr_op->get_input_size() == 5) {
                    data_indices = rg.make<v8::Slice>(data_indices,
                                                      curr_op->input_value(2),
                                                      curr_op->input_value(3),
                                                      curr_op->input_value(4));
                } else {
                    return false;
                }
            } else if (ov::as_type_ptr<GatherAssign>(curr_op)) {
                data_indices = rg.make<v8::Gather>(data_indices, curr_op->input_value(2), curr_op->input_value(3));
            } else {
                return false;
            }
        }

        // Scatter in flattened tensor with indices and flattened data to be inserted
        auto data_to_insert_into_1d = rg.make<v1::Reshape>(data_to_insert_into, neg_one_1d, false);
        auto data_indices_1d = rg.make<v1::Reshape>(data_indices, scattering_shape, false);
        auto to_be_inserted_data_1d = rg.make<v1::Reshape>(base_op->input_value(1), neg_one_1d, false);
        auto updated_data_1d =
            rg.make<v3::ScatterNDUpdate>(data_to_insert_into_1d, data_indices_1d, to_be_inserted_data_1d);

        // Reshape to initial shape
        auto res_node = rg.make<v1::Reshape>(updated_data_1d, input_shape, false);
        copy_runtime_info_and_name(base_op, rg.get());
        start_op->output(0).replace(res_node);

        return true;
    };

    auto m =
        std::make_shared<ov::pass::pattern::Matcher>(reverse_op, "ov::frontend::pytorch::pass::ReversepropResolver");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
