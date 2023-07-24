// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_compress.hpp"

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::opset11;
using namespace ov::pass;
using namespace ov::intel_gna::pass;

namespace {

inline std::vector<size_t> fix_indexes(const std::vector<size_t>& ids) {
    std::vector<size_t> ids_fixed(ids.size());
    std::iota(ids_fixed.begin(), ids_fixed.end(), 0);
    stable_sort(ids_fixed.begin(), ids_fixed.end(), [&ids](size_t a, size_t b) {
        return ids[a] < ids[b];
    });
    return ids_fixed;
}

}  // namespace

TransposeCompress::TransposeCompress() {
    MATCHER_SCOPE(TransposeCompress);

    auto transpose_const = pattern::wrap_type<Constant>();
    auto transpose =
        pattern::wrap_type<Transpose>({pattern::any_input(), transpose_const}, [](const ov::Output<ov::Node>& node) {
            return !limitations::Limitations::is_transpose_supported(node.get_node_shared_ptr());
        });

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto transpose_order = as_type_ptr<Constant>(pattern_map.at(transpose_const).get_node_shared_ptr());
        const auto transpose_node = pattern_map.at(transpose).get_node_shared_ptr();
        const ov::Shape& shape = transpose_node->get_input_shape(0);
        const ov::Shape& shape_out = transpose_node->get_output_shape(0);
        ov::AxisVector axis = transpose_order->get_axis_vector_val();
        ov::AxisVector axis_compressed = {};
        ov::Shape shape_compressed_out = {};
        for (const size_t& axis : axis) {
            if (axis_compressed.empty() || (axis - axis_compressed.back()) != 1) {
                axis_compressed.push_back(axis);
                shape_compressed_out.push_back(shape[axis]);
            } else {
                shape_compressed_out.back() *= shape[axis];
            }
        }
        // check that compressing is required
        if (axis.size() == axis_compressed.size()) {
            return false;
        }

        // correct fused indexes, e.g. (2, 0, 3) -> (1, 0, 2)
        ov::AxisVector axis_fused_fixed = fix_indexes(axis_compressed);
        size_t fused_sz = axis_fused_fixed.size();
        // Restore input shape
        ov::Shape shape_fused_in(fused_sz);
        for (size_t i = 0; i < fused_sz; ++i) {
            shape_fused_in[i] = shape_compressed_out[axis_fused_fixed[i]];
        }

        if (!limitations::Limitations::is_transpose_supported(shape_fused_in)) {
            return false;
        }

        // Reshape in
        auto reshape_in_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_fused_in.size()}, shape_fused_in);
        auto reshape_in = std::make_shared<Reshape>(transpose_node->input_value(0), reshape_in_const, false);
        // Transpose
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{axis_fused_fixed.size()}, axis_fused_fixed);
        auto transpose = std::make_shared<Transpose>(reshape_in, transpose_const);
        // Reshape out
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out = std::make_shared<Reshape>(transpose, reshape_out_const, false);
        //
        ov::replace_output_update_name(transpose_node->output(0), reshape_out->output(0));
        ov::copy_runtime_info({transpose_node, transpose_order},
                              {transpose, transpose_const, reshape_in, reshape_in_const});

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose, matcher_name);
    this->register_matcher(m, callback);
}
