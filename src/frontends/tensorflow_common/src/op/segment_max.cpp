// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include "common_op_table.hpp"
#include "utils.hpp"

namespace ov::frontend::tensorflow::op {
OutputVector translate_segment_max_op(const NodeContext& node) {
    default_op_checks(node, 2, {"SegmentMax"});
    auto data = node.get_input(0);
    auto segment_ids = node.get_input(1);
    auto res = std::make_shared<ov::op::v16::SegmentMax>(data, segment_ids, ov::op::FillMode::ZERO);
    set_node_name(node.get_name(), res);
    return {res};
}
}  // namespace ov::frontend::tensorflow::op
