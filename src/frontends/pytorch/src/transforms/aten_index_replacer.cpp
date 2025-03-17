// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "aten_index_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

AtenIndexToSelect::AtenIndexToSelect() {
    auto index_op = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto index_op = cast_fw_node(m.get_match_root(), "aten::index");
        if (!index_op) {
            return false;
        }
        ov::pass::NodeRegistry rg;
        auto input_node = index_op->input_value(0);
        auto indicies = index_op->input_value(1).get_node_shared_ptr();
        auto list_indicies = cast_fw_node(indicies, "prim::ListConstruct");
        if (list_indicies) {
            auto ids = list_indicies->input_values();
            auto rank = input_node.get_partial_shape().rank();
            // index transformation supports only tensors with static rank
            ov::Output<ov::Node> new_output;
            bool use_input_as_output = true;
            if (!index_tensor_on_list(rg, input_node, ids, rank, new_output, use_input_as_output)) {
                add_exception_to_fw_node(index_op, "aten::index: dynamic rank for aten::index input is not supported.");
                return false;
            }
            if (use_input_as_output) {
                index_op->output(0).replace(index_op->get_input_source_output(0));
                return true;
            }
            copy_runtime_info_and_name(index_op, rg.get());
            replace_node(index_op, new_output.get_node_shared_ptr());
            return true;
        } else {
            auto const_input = cast_fw_node(indicies, "prim::Constant");

            if (const_input) {
                // index is None, stay input as is
                const auto& attrs = const_input->get_attrs();
                if (attrs.find("none_value") != attrs.end()) {
                    index_op->output(0).replace(index_op->get_input_source_output(0));
                    return true;
                }
            }
            auto index_dtype = indicies->get_output_element_type(0);
            if (index_dtype == element::boolean || index_dtype == element::u8) {
                auto nonzero = rg.make<v3::NonZero>(indicies);
                auto input_order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
                auto masked_id = rg.make<v1::Transpose>(nonzero, input_order);
                auto gather = rg.make<v8::GatherND>(input_node, masked_id);
                copy_runtime_info_and_name(index_op, rg.get());
                replace_node(index_op, gather);
                return true;
            }
            if (index_dtype != element::i32) {
                indicies = rg.make<ov::op::v0::Convert>(indicies, element::i32);
            }
            auto dim = v0::Constant::create(element::i32, Shape{}, {0});
            auto gather = rg.make<v8::Gather>(input_node, indicies, dim);
            copy_runtime_info_and_name(index_op, rg.get());
            replace_node(index_op, gather);
            return true;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(index_op, "ov::frontend::pytorch::pass::AtenIndexToSelect");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
