//// Copyright (C) 2018-2025 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#include "openvino/pass/pattern/op/op.hpp"
//
//#include "openvino/pass/pattern/op/label.hpp"
//#include "openvino/pass/pattern/op/wrap_type.hpp"
//#include "openvino/op/constant.hpp"
//#include "openvino/util/common_util.hpp"
//
// namespace ov::pass::pattern {
//    PatternOp::operator ov::Output<ov::Node>() const {
//        return get_output();
//    }
//
//    ov::Output<ov::Node> PatternOp::get_output() const {
//        if (output_idx >= 0)
//            return op->output(output_idx);
//        return op->get_default_output();
//    }
//
//    PatternOp::PatternOp(const Output<Node> &out)
//            : op(out.get_node_shared_ptr()),
//              output_idx(out.get_index()) {}
//
//    PatternOp::PatternOp() {
//        op = any_input();
//    }
//
//    PatternOp::PatternOp(ov::Rank rank) {
//        op = any_input(rank_equals(rank));
//    }
//
//    PatternOp::PatternOp(std::string value_notation) {
//        op = wrap_type<ov::op::v0::Constant>(value_matches(value_notation));
//    }
//
//    PatternOp::PatternOp(int v) : PatternOp(std::to_string(v)) {}
//    PatternOp::PatternOp(float v) : PatternOp(std::to_string(v)) {}
//    PatternOp::PatternOp(double v) : PatternOp(std::to_string(v)) {}
//    PatternOp::(long long v) : PatternOp(std::to_string(v)) {}
//
//    PatternOp::PatternOp(std::initializer_list<int> v) : PatternOp(ov::util::join(v, ",")) {}
//    PatternOp::PatternOp(std::initializer_list<float> v) : PatternOp(ov::util::join(v, ",")) {}
//    PatternOp::PatternOp(std::initializer_list<double> v) : PatternOp(ov::util::join(v, ",")) {}
//    PatternOp::PatternOp(std::initializer_list<long> v) : PatternOp(ov::util::join(v, ",")) {}
//}