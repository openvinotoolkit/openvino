// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_shape_of.hpp"

#include "itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

ov::pass::transpose_sinking::TSShapeOfForward::TSShapeOfForward() {
    MATCHER_SCOPE(TSShapeOfForward);

    create_pattern<op::util::ShapeOfBase>();
    auto sinking_transformation = [=](const std::shared_ptr<Node>& main_node,
                                      const utils::TransposeInputsInfo& transpose_info) -> bool {
        main_node->input(0).replace_source_output(transpose_info.transpose->input_value(0));
        auto shape_of_consumers = main_node->get_output_target_inputs(0);
        const auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        const auto gather = std::make_shared<op::v8::Gather>(main_node, transpose_info.transpose_const, axis);
        for (auto& input : shape_of_consumers)
            input.replace_source_output(gather);
        copy_runtime_info(main_node, gather);
        utils::SwapOutputNames(main_node->output(0), gather->output(0));
        utils::SwapFriendlyNames(main_node, gather);

        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}
