// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_slicescatter.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/slice_scatter.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertSliceScatter::ConvertSliceScatter() {
    MATCHER_SCOPE(ConvertSliceScatter);

    const auto& slicescatter = pattern::wrap_type<ov::op::v15::SliceScatter>();

    const matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto& slice_node = ov::as_type_ptr<ov::op::v15::SliceScatter>(m.get_match_root());
        if (!slice_node || transformation_callback(slice_node)) {
            return false;
        }
        NodeRegistry node_registry;
        const auto& const_0 = node_registry.make<ov::op::v0::Constant>(ov::element::i64, Shape{}, 0);
        const auto& const_1 = node_registry.make<ov::op::v0::Constant>(ov::element::i64, Shape{}, 1);
        const auto& const_1d_neg_1 =
            node_registry.make<ov::op::v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{-1});
        const auto& const_scatter_indices_shape =
            node_registry.make<ov::op::v0::Constant>(ov::element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
        const auto& data_shape = node_registry.make<ov::op::v3::ShapeOf>(slice_node->input_value(0), ov::element::i64);
        const auto& num_elements_data = node_registry.make<ov::op::v1::ReduceProd>(data_shape, const_0, false);
        const auto& data_indices_flatten =
            node_registry.make<ov::op::v4::Range>(const_0, num_elements_data, const_1, ov::element::i64);
        const auto& full_data_indices =
            node_registry.make<ov::op::v1::Reshape>(data_indices_flatten, data_shape, false);
        std::shared_ptr<ov::op::v8::Slice> slice_indices;
        if (slice_node->get_input_size() == 5) {
            slice_indices = node_registry.make<ov::op::v8::Slice>(full_data_indices,
                                                                  slice_node->input_value(2),
                                                                  slice_node->input_value(3),
                                                                  slice_node->input_value(4));
        } else {
            slice_indices = node_registry.make<ov::op::v8::Slice>(full_data_indices,
                                                                  slice_node->input_value(2),
                                                                  slice_node->input_value(3),
                                                                  slice_node->input_value(4),
                                                                  slice_node->input_value(5));
        }
        const auto& slice_indices_flatten =
            node_registry.make<ov::op::v1::Reshape>(slice_indices, const_scatter_indices_shape, false);
        const auto& updates_flatten =
            node_registry.make<ov::op::v1::Reshape>(slice_node->input_value(1), const_1d_neg_1, false);
        const auto& data_flatten =
            node_registry.make<ov::op::v1::Reshape>(slice_node->input_value(0), const_1d_neg_1, false);
        const auto& output_flatten =
            node_registry.make<ov::op::v3::ScatterNDUpdate>(data_flatten, slice_indices_flatten, updates_flatten);
        const auto& output = node_registry.make<ov::op::v1::Reshape>(output_flatten, data_shape, false);

        output->set_friendly_name(slice_node->get_friendly_name());
        copy_runtime_info(slice_node, node_registry.get());
        replace_node(slice_node, output);

        return true;
    };

    const auto& m = std::make_shared<pattern::Matcher>(slicescatter, matcher_name);
    this->register_matcher(m, callback);
}
