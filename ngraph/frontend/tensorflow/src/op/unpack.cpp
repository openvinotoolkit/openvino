// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>
//#include "node_context.hpp"

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

static Status TranslateUnpackOp(const TFNodeDecoder* op,
                                const std::vector<const ngraph::frontend::tf::detail::TensorWrapper*>&,
                                Builder::OpMap& ng_op_map) {
    TF_RETURN_IF_ERROR(ValidateInputCount(op, 1));

    Output<Node> ng_input;
    TF_RETURN_IF_ERROR(GetInputNode(ng_op_map, op, 0, ng_input));
    int32_t tf_axis;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "axis", &tf_axis));
    int32_t num_outputs;
    TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "num", &num_outputs));

    auto input_shape = ng_input.get_shape();
    auto rank = input_shape.size();
    for (int i = 0; i < num_outputs; ++i) {
        std::vector<int64_t> begin(rank, 0);
        std::vector<int64_t> end(rank, 0);
        begin[tf_axis] = i;
        end[tf_axis] = i + 1;
        auto ng_begin = ConstructNgNode<Constant>(
                node.get_name(), element::i64, Shape{begin.size()}, begin);
        auto ng_end = ConstructNgNode<Constant>(node.get_name(), element::i64,
                                                       Shape{end.size()}, end);
        std::vector<int64_t> begin_mask(rank, 1);
        begin_mask[tf_axis] = 0;
        std::vector<int64_t> end_mask(rank, 1);
        end_mask[tf_axis] = 0;
        std::vector<int64_t> new_axis_mask(rank, 0);
        std::vector<int64_t> shrink_axis_mask(rank, 0);
        shrink_axis_mask[tf_axis] = 1;
        auto slice = ConstructNgNode<StridedSlice>(
                node.get_name(), ng_input, ng_begin, ng_end, begin_mask, end_mask,
                new_axis_mask, shrink_axis_mask);
        SaveNgOp(ng_op_map, node.get_name(), slice);
    }
    return Status::OK();
}
}
}

#endif