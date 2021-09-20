// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0
namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateSliceOp(
        const NodeContext& node) {
    Output<Node> ng_input, ng_begin, ng_size;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_input, ng_begin, ng_size));

    std::vector<int64_t> begin_vec;
    std::vector<int64_t> size_vec;
    TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 1, static_input_map, &begin_vec));
    TF_RETURN_IF_ERROR(GetStaticInputVector(ng_op_map, op, 2, static_input_map, &size_vec));

    if (begin_vec.size() != size_vec.size())
        return errors::InvalidArgument(
                "Cannot translate slice op: size of begin = " + to_string(begin_vec.size()) +
                ", size of size_vec = " + to_string(size_vec.size()) + ". Expected them to match.");

    NGRAPH_VLOG(3) << "Begin input for Slice: " << join(begin_vec);
    NGRAPH_VLOG(3) << "Size input for Slice: " << join(size_vec);

    std::vector<int64_t> end_vec(begin_vec.size());
    const auto ng_input_shape = ng_input.get_shape();
    stringstream err_stream;
    string err_msg;
    for (size_t i = 0; i < size_vec.size(); i++) {
        if (size_vec[i] != -1) {
            end_vec[i] = begin_vec[i] + size_vec[i];
        } else {
            // support -1 for size_vec, to the end of the tensor
            end_vec[i] = ng_input_shape[i];
        }

        // check for this condition: 0 <= begin[i] <= begin[i] + size[i] <= Di
        if (0 > begin_vec[i])
            err_stream << "lower < 0: " << begin_vec[i]
                       << ". It should have been positive.\n";
        if (begin_vec[i] > end_vec[i])
            err_stream << "upper < lower: upper = " << end_vec[i]
                       << ", lower = " << begin_vec[i] << "\n";
        if (begin_vec[i] > ng_input_shape[i])
            err_stream << "dim < upper: dim = " << ng_input_shape[i]
                       << ", upper = " << end_vec[i] << "\n";

        err_msg = err_stream.str();
        if (!err_msg.empty())
            return errors::InvalidArgument("Cannot translate slice op at position " +
                                           to_string(i) + " of " + to_string(size_vec.size()) +
                                           ". The reasons are:\n" + err_msg);
    }

    auto begin = ConstructNgNode<opset::Constant>(
            node.get_name(), element::i64, Shape{begin_vec.size()}, begin_vec);
    auto end = ConstructNgNode<opset::Constant>(
            node.get_name(), element::i64, Shape{end_vec.size()}, end_vec);

    SaveNgOp(ng_op_map, node.get_name(),
             ConstructNgNode<opset::StridedSlice>(node.get_name(), ng_input, begin,
                                                  end, std::vector<int64_t>{},
                                                  std::vector<int64_t>{}));
    return Status::OK();
}
}
}

#endif