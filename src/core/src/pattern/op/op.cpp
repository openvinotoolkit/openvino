// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/op.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::pass::pattern {
PatternOp::operator ov::Output<ov::Node>() const {
    return get_output();
}

ov::Output<ov::Node> PatternOp::get_output() const {
    if (output_idx >= 0)
        return op->output(output_idx);
    return op->get_default_output();
}

PatternOp::PatternOp(const Output<Node>& out) : op(out.get_node_shared_ptr()), output_idx(out.get_index()) {}

PatternOp::PatternOp(const std::shared_ptr<Node>& op) : op(op) {}

PatternOp::PatternOp(ov::Rank rank) {
    op = any_input(rank_equals(rank));
}

PatternOp::PatternOp(std::string value_notation) {
    op = wrap_type<ov::op::v0::Constant>(value_matches(value_notation));
}

PatternOp::PatternOp(const char* value_notation) : PatternOp(std::string(value_notation)) {}

PatternOp::PatternOp(int v) : PatternOp(std::to_string(v)) {}
PatternOp::PatternOp(float v) : PatternOp(std::to_string(v)) {}
PatternOp::PatternOp(double v) : PatternOp(std::to_string(v)) {}
PatternOp::PatternOp(long long v) : PatternOp(std::to_string(v)) {}

PatternOp::PatternOp(std::initializer_list<const char*>&& v) : PatternOp(ov::util::join(v)) {}
PatternOp::PatternOp(std::initializer_list<const std::string>&& v) : PatternOp(ov::util::join(v)) {}
PatternOp::PatternOp(std::initializer_list<const int>&& v) : PatternOp(ov::util::join(v)) {}
PatternOp::PatternOp(std::initializer_list<const float>&& v) : PatternOp(ov::util::join(v)) {}
PatternOp::PatternOp(std::initializer_list<const double>&& v) : PatternOp(ov::util::join(v)) {}
PatternOp::PatternOp(std::initializer_list<const long long>&& v) : PatternOp(ov::util::join(v)) {}

PatternOps::PatternOps() : data{} {}
PatternOps::PatternOps(const std::shared_ptr<Node>& op) : data{PatternOp(op)} {}
PatternOps::PatternOps(const Output<Node>& out) : data{PatternOp(out)} {}

PatternOps::PatternOps(const OutputVector& outputs) : data{outputs.begin(), outputs.end()} {}
PatternOps::PatternOps(std::initializer_list<PatternOp>&& outputs) : data(outputs) {}

PatternOps::operator ov::OutputVector() const {
    OutputVector args;
    for (auto& in : data)
        args.push_back(in.get_output());
    return args;
}
}  // namespace ov::pass::pattern