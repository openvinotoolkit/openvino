// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_bias_add_op(const NodeContext& node) {
    default_op_checks(node, 2, {"BiasAdd"}, true);
    auto value = node.get_input(0);
    auto bias = node.get_input(1);

    auto complex_type_mark_value = as_type_ptr<ComplexTypeMark>(value.get_node_shared_ptr());
    auto complex_type_mark_bias = as_type_ptr<ComplexTypeMark>(bias.get_node_shared_ptr());
    auto complex_type_inputs = (complex_type_mark_value || complex_type_mark_bias) ? true : false;
    // validations prior to processing
    if (complex_type_inputs) {
        // extractions for complex processing
        bias = complex_type_mark_bias->get_data();
        value = complex_type_mark_value->get_data();
    }

    // retrieve optional attributes
    std::string data_format = node.get_attribute<std::string>("data_format", "NHWC");
    TENSORFLOW_OP_VALIDATION(node,
                             data_format == "NHWC" || data_format == "NCHW",
                             "BiasAdd data format is neither NHWC nor NCHW.");

    Output<Node> bias_reshaped = bias;

    // in case NCHW layout bias must be reshaped to have a shape (1, C, 1, ...)
    // for further correct use of Add operation
    if (data_format == "NCHW") {
        // TODO: add support for dynamic rank in case NCHW layout
        auto value_shape = value.get_partial_shape();
        TENSORFLOW_OP_VALIDATION(node,
                                 value_shape.rank().is_static(),
                                 "Value of dynamic rank for BiasAdd in NCHW layout is not supported.");
        auto value_rank = complex_type_inputs ? value_shape.rank().get_length() - 1 : value_shape.rank().get_length();

        std::vector<int64_t> axes_unsqueeze;
        for (int64_t dim_ind = 0; dim_ind < value_rank; ++dim_ind) {
            if (dim_ind != 1) {
                axes_unsqueeze.push_back(dim_ind);
            }
        }
        auto axes_unsqueeze_node =
            make_shared<v0::Constant>(element::i64, Shape{axes_unsqueeze.size()}, axes_unsqueeze);
        bias_reshaped = make_shared<v0::Unsqueeze>(bias, axes_unsqueeze_node);
    }

    auto res = make_shared<v1::Add>(value, bias_reshaped);
    set_node_name(node.get_name(), res);

    if (complex_type_inputs) {
        auto complex_reshape = make_shared<ComplexTypeMark>(res, complex_type_mark_value->get_complex_part_type());
        return {complex_reshape->output(0)};
    }
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
