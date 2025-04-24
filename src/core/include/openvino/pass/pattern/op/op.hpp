// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"

namespace ov::pass::pattern {
// A glue/syntax-sugar type which allows more types to be used as input to pattern operations
struct OPENVINO_API PatternOp {
private:
    std::shared_ptr<ov::Node> op;
    int64_t output_idx = -1;

public:
    operator ov::Output<ov::Node>() const;
    ov::Output<ov::Node> get_output() const;

    PatternOp(const Output<Node>& out);

    template <typename T, typename std::enable_if_t<std::is_base_of_v<ov::Node, T>>* = nullptr>
    PatternOp(const std::shared_ptr<T>& op) : op(std::dynamic_pointer_cast<ov::Node>(op)) {}

    PatternOp(const std::shared_ptr<Node>& op);
    PatternOp(ov::Rank rank);

    // Constant matching
    PatternOp(const char* value_notation);
    PatternOp(std::string value_notation);
    PatternOp(int v);
    PatternOp(float v);
    PatternOp(double v);
    PatternOp(long long v);

    PatternOp(std::initializer_list<const char*>&& v);
    PatternOp(std::initializer_list<const std::string>&& v);
    PatternOp(std::initializer_list<const int>&& v);
    PatternOp(std::initializer_list<const float>&& v);
    PatternOp(std::initializer_list<const double>&& v);
    PatternOp(std::initializer_list<const long long>&& v);
};

// Syntax-sugar type for pattern operators to consume all the different ways to pass containter of inputs with use of
// PatternOp
struct OPENVINO_API PatternOps {
private:
    std::vector<PatternOp> data;

public:
    PatternOps();

    // single element
    template <typename T, typename std::enable_if_t<std::is_constructible_v<PatternOp, T>>* = nullptr>
    PatternOps(const T& in) : data{PatternOp(in)} {};
    PatternOps(const std::shared_ptr<Node>&);
    PatternOps(const Output<Node>&);

    // multi-element
    PatternOps(const OutputVector&);
    PatternOps(std::initializer_list<pattern::PatternOp>&&);

    explicit operator ov::OutputVector() const;
};

}  // namespace ov::pass::pattern