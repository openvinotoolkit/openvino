//// Copyright (C) 2018-2025 Intel Corporation
//// SPDX-License-Identifier: Apache-2.0
////
//
//#pragma once
//
//#include "openvino/pass/pattern/op/predicate.hpp"
//
// namespace ov::pass::pattern {
//// A glue/syntax-sugar type which allows more types to be used as input to pattern operations
// struct PatternOp {
//     std::shared_ptr<ov::Node> op;
//     size_t output_idx;
//
//     operator ov::Output<ov::Node>() const;
//     ov::Output<ov::Node> get_output() const;
//
//     PatternOp();
//     PatternOp(const Output<Node> &out);
//
//     template <typename TPredicate,
//               typename std::enable_if_t<std::is_constructible_v<op::Predicate, TPredicate>>>* = nullptr>
//     PatternOp(PredicateT pred) {
//         op = any_input(pred);
//     }
//
//     PatternOp(ov::Rank rank);
//
//     // Constant matching
//     PatternOp(std::string value_notation);
//     PatternOp(int v);
//     PatternOp(float v);
//     PatternOp(double v);
//     PatternOp(long long v);
//     PatternOp(std::initializer_list<int> v);
//     PatternOp(std::initializer_list<float> v);
//     PatternOp(std::initializer_list<double> v);
//     PatternOp(std::initializer_list<long> v);
// };
// }