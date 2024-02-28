// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"


using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_sub_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Sub"}, true);
    auto lhs = node.get_input(0);
    auto rhs = node.get_input(1);

    auto result = make_shared<v1::Subtract>(lhs, rhs);

    auto complex_type_mark_lhs = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto complex_type_mark_rhs = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());

    if (complex_type_mark_lhs || complex_type_mark_rhs) {
        
        lhs = complex_type_mark_lhs->input_value(0);
        rhs = complex_type_mark_rhs->input_value(0);

        element::Type complex_part_type_lhs = complex_type_mark_lhs->get_complex_part_type();
        element::Type complex_part_type_rhs = complex_type_mark_rhs->get_complex_part_type();
        
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{}, 1);

        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        auto lhs_real = make_shared<v8::Gather>(lhs, gather_index_real, minus_one)->output(0);
        auto lhs_imag = make_shared<v8::Gather>(lhs, gather_index_imag, minus_one)->output(0);
        
        auto rhs_real = make_shared<v8::Gather>(rhs, gather_index_real, minus_one)->output(0);
        auto rhs_imag = make_shared<v8::Gather>(rhs, gather_index_imag, minus_one)->output(0);

        // result_real = lhs_real - rhs_real
        auto result_real = make_shared<v1::Subtract>(lhs_real, rhs_real);
        // result_imag = lhs_imag - rhs_imag
        auto result_imag = make_shared<v1::Subtract>(lhs_imag, rhs_imag);

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(result_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(result_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type_lhs);
        return {complex_result};
    }
    
    set_node_name(node.get_name(), result);
    
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
