// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_tensorarray.hpp"

#include <ngraph/log.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>

#include "default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "internal/op/while.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/label.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace frontend::paddle::op::default_opset;

// Transform pattern "TensorArrayLength->TensorArrayWrite" to OV concat, which
// will append to the end of array after unsqueeze along axis 0.
ov::frontend::paddle::pass::TransformTensorArray::TransformTensorArray(std::vector<std::shared_ptr<Model>> functions) {
    const auto shape_label = ngraph::pattern::wrap_type<ShapeOf>();
    const auto length_label = ngraph::pattern::wrap_type<StridedSlice>(
        {shape_label, ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto write_label =
        ngraph::pattern::wrap_type<ov::op::internal::TensorArrayWrite>({ngraph::pattern::any_input(), length_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto& write_node = opsMap.at(write_label).get_node_shared_ptr();
        const auto& shape_node = opsMap.at(shape_label).get_node_shared_ptr();
        if (!write_node || !shape_node)
            return false;
        const auto& new_item = write_node->get_input_node_shared_ptr(0);
        const auto& list = shape_node->get_input_node_shared_ptr(0);
        const auto& new_item_unsqueeze = std::make_shared<Unsqueeze>(
            new_item->output(0),
            Constant::create(element::i32, {1}, {0}));  // unsqueeze in order to handyfully slice a tensorarray
        // remove TensorArrayLength->TensorArrayWrite
        const auto concat = std::make_shared<Concat>(OutputVector{list->output(0), new_item_unsqueeze->output(0)}, 1);
        // prevent to remove concating zero-tensor
        ov::pass::disable_remove_concat_zerodim_input(concat);

        replace_node(write_node, concat);
        concat->set_friendly_name(write_node->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(write_label, "tensorarray");
    this->register_matcher(m, callback);
}