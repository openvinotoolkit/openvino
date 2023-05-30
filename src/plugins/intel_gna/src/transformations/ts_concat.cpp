// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/ts_concat.hpp"

#include <openvino/cc/ngraph/itt.hpp>

#include "../debug_new_pass.hpp"
#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::limitations;

namespace {

size_t GetNumFirstOneDims(const ov::Shape& shape) {
    size_t count = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != 1)
            break;
        ++count;
    }
    return count;
}

bool is_sinked(const Output<Node>& output) {
    auto concat_node = ov::as_type_ptr<Concat>(output.get_node_shared_ptr());

    const Shape concat_output_shape = concat_node->get_output_shape(0);
    const int64_t axis = concat_node->get_concatenation_axis();
    if (GetNumFirstOneDims(concat_output_shape) != axis)
        return false;

    for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
        auto concat_input = concat_node->input_value(i);
        auto transpose = ov::as_type_ptr<Transpose>(concat_input.get_node_shared_ptr());
        if (transpose && !Limitations::is_transpose_supported(transpose))
            return true;
    }

    return false;
}

std::vector<size_t> CreateGatherIndices(const ov::Shape& input_shape, const ov::Shape& order) {
    if (input_shape.size() < 2 || input_shape.size() > 4) {
        THROW_GNA_EXCEPTION << "Usupported shape size: " << input_shape.size();
    }

    ov::Shape input_shape_4d = input_shape;
    ov::Shape order_4d = order;
    // Just to simplify the code we transform all shapes to 4d by adding 1 dimentions at the end
    while (input_shape_4d.size() < 4) {
        input_shape_4d.push_back(1);
        order_4d.push_back(order_4d.size());
    }
    ov::Shape output_shape_4d = transpose_shape(input_shape_4d, order_4d);

    // common case when shape is 4d
    std::vector<size_t> xyz_4d = {input_shape_4d[3] * input_shape_4d[2] * input_shape_4d[1],
                                  input_shape_4d[3] * input_shape_4d[2],
                                  input_shape_4d[3],
                                  1};

    std::vector<size_t> xyz = transpose_shape(xyz_4d, order_4d);
    std::vector<size_t> gather_order;

    for (size_t n = 0; n < output_shape_4d[0]; ++n) {
        for (size_t i = 0; i < output_shape_4d[1]; ++i) {
            for (size_t j = 0; j < output_shape_4d[2]; ++j) {
                for (size_t k = 0; k < output_shape_4d[3]; ++k) {
                    gather_order.push_back(n * xyz[0] + i * xyz[1] + j * xyz[2] + k * xyz[3]);
                }
            }
        }
    }

    return gather_order;
}

}  // namespace

TSConcatForward::TSConcatForward() {
    MATCHER_SCOPE(TSConcatForward);

    auto concat_node_label = wrap_type<Concat>(is_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& concat_node_output = pattern_to_output.at(concat_node_label);
        auto concat_node = as_type_ptr<Concat>(concat_node_output.get_node_shared_ptr());

        std::vector<AxisVector> gather_indices_vecs;
        OutputVector concat_inputs;
        for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
            auto concat_input = concat_node->input_value(i);
            auto transpose = ov::as_type_ptr<Transpose>(concat_input.get_node_shared_ptr());
            const bool is_transposed_input = transpose && !Limitations::is_transpose_supported(transpose);

            auto input_node_output = concat_input;
            if (is_transposed_input)
                input_node_output = concat_input.get_node_shared_ptr()->input_value(0);

            const Shape& input_shape = input_node_output.get_shape();
            const size_t input_dims =
                std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<Shape::value_type>());
            Shape reshape_shape = {1, input_dims};
            auto reshape_input_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{2}, reshape_shape);
            auto reshape_input = std::make_shared<Reshape>(input_node_output, reshape_input_const, false);
            concat_inputs.push_back(reshape_input->output(0));

            if (is_transposed_input) {
                auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
                if (!transpose_const)
                    return false;
                auto gather_indices_value =
                    CreateGatherIndices(transpose->get_input_shape(0), transpose_const->get_axis_vector_val());
                gather_indices_vecs.push_back(gather_indices_value);
            } else {
                const Shape& input_shape = concat_input.get_shape();
                const size_t input_dims =
                    std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<Shape::value_type>());
                std::vector<size_t> indices(input_dims);
                std::iota(indices.begin(), indices.end(), 1);
                gather_indices_vecs.push_back(indices);
                continue;
            }
        }

        auto concat_new = std::make_shared<Concat>(concat_inputs, 1);

        auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        std::vector<size_t> gather_indices_value;
        {
            size_t shift = 0;
            for (const auto& indices : gather_indices_vecs) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    gather_indices_value.push_back(indices[i] + shift);
                }
                shift += indices.size();
            }
        }
        auto gather_indices =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
        auto gather = std::make_shared<Gather>(concat_new, gather_indices, gather_axis);

        auto reshape_output_const_new = std::make_shared<Constant>(ov::element::i64,
                                                                   ov::Shape{concat_node->get_output_shape(0).size()},
                                                                   concat_node->get_output_shape(0));
        auto reshape_output_new = std::make_shared<Reshape>(gather, reshape_output_const_new, false);

        ov::replace_node_update_name(concat_node, reshape_output_new);
        return true;
    };

    auto m = std::make_shared<Matcher>(concat_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
