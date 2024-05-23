// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "utils.hpp"

#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/reshape.hpp"

#include <fstream>

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

// DIFERENT STRA: Create padding tensor based on the param and get the expected shape. update the index based on k.
    
// https://github.com/tensorflow/tensorflow/blob/84d053187cb80d975ef2b9684d4b61981bca0c41/tensorflow/core/kernels/linalg/matrix_diag_op.cc#L151
// https://github.com/tensorflow/tensorflow/blob/84d053187cb80d975ef2b9684d4b61981bca0c41/tensorflow/core/kernels/linalg/matrix_diag_op.cc#L256
// https://github.com/tensorflow/tensorflow/blob/84d053187cb80d975ef2b9684d4b61981bca0c41/tensorflow/core/kernels/linalg/matrix_diag_op.cc#L330
OutputVector translate_matrix_diag_v3_op(const NodeContext& node) {

    std::ofstream logFile("debug.log", std::ios_base::app);
    logFile << std::endl;
    logFile << "START" << std::endl;

    default_op_checks(node, 1, {"MatrixDiagV3"});

    // The translation of MatrixDiag to OpenVINO opset relies on padding of input tensor with zeros,
    // reshape to a special form and cutting of unneeded padding part.
    // Here is a basic idea described by an example,
    // let us have a tensor [1, 2, 3] and generate padding tensor of zeros with a shape [3, 3]. k = 0
    // Concatenate input tensor with padding and get the following:
    // [[1, 0, 0, 0]
    //  [2, 0, 0, 0]
    //  [3, 0, 0, 0]] of shape [3, 4]
    // Reshape to tensor of a shape [12] equal to [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
    // Cut off last 3 elements and get [1, 0, 0, 0, 2, 0, 0, 0, 3] and reshape to [3, 3]
    // This idea is generalized to higher rank tensors
    // diagonal is the single input to MatrixDiag operation and has a shape [I, J, ..., M, N]

    // k = 1
    // Concatenate input tensor with padding and get the following:
    // [[0, 1, 0, 0, 0]
    //  [0, 2, 0, 0, 0]
    //  [0, 3, 0, 0, 0]
    //  [0, 0, 0, 0, 0]] of shape [4, 5]
    // Reshape to tensor of a shape [20] equal to [0, 1, 0, 0, \\ 0, 0, 2, 0, \\ 0, 0, 0, 3, \\ 0, 0, 0, 0, \\ 0, 0, 0, 0]
    // Cut off last 5 elements and get [0, 1, 0, 0, \\ 0, 0, 2, 0, \\ 0, 0, 0, 3, \\ 0, 0, 0, 0] and reshape to [4, 4]
    auto diagonal = node.get_input(0);
    logFile << "Shape of diagonal: " << diagonal.get_shape().to_string() << std::endl;
    auto k = node.get_attribute<int64_t>("k", 0); // get an int instead of dynamic size node
    auto num_rows = node.get_attribute<int64_t>("num_rows", -1); 
    auto num_cols = node.get_attribute<int64_t>("num_cols", -1); 
    auto padding_value_ = node.get_attribute<int64_t>("padding_value", 0);
    // auto align = node.get_input(5);
    auto diagonal_type = diagonal.get_element_type();

    logFile << k << num_cols << num_rows << padding_value_ << std::endl;
    auto padding_value = make_shared<v0::Constant>(diagonal_type, Shape{}, std::vector<int64_t>{padding_value_}); 

    // 1. unsqueeze to have at least three rank input of a shape [1, I, J, ..., M, N, 1]
    //    because dimensions [I, J, ..., M] can be absent
    auto unsqueeze_axis = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{0, -1});
    auto unsqueeze_diag = make_shared<v0::Unsqueeze>(diagonal, unsqueeze_axis);

    // 2. compute a size of the last dimension of the diagonal input of a shape [I, J, ..., M, N],
    //    i.e. N that will be diagonalized
    auto unsqueeze_diag_shape = make_shared<v3::ShapeOf>(unsqueeze_diag);
    auto last_dim =
        make_shared<v1::StridedSlice>(unsqueeze_diag_shape,
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-2}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                      std::vector<int64_t>({0}),
                                      std::vector<int64_t>({0}));

    logFile << "Shape of unsqueeze_diag_shape: " << unsqueeze_diag_shape->get_shape().to_string() << std::endl;

    logFile << "Shape of last_dim: " << last_dim->get_shape().to_string() << std::endl;

    // 3. generate a tensor of zeros of a shape [1, I, J, ..., M, N, N]
    auto diag_shape = make_shared<v3::ShapeOf>(diagonal);
    auto one_dim = make_shared<v0::Constant>(last_dim->get_element_type(), Shape{1}, std::vector<int64_t>{1});
    auto padding_shape = make_shared<v0::Concat>(OutputVector({one_dim, diag_shape, last_dim}), 0);
    auto padding = make_shared<v3::Broadcast>(padding_value, padding_shape);

    logFile << "Shape of padding: " <<padding->get_shape().to_string() << std::endl;

    // 4. concatenate to get input tensor with padding of a shape [1, I, J, ..., M, N, N + 1]
    auto zero_padded_diag = make_shared<v0::Concat>(OutputVector({unsqueeze_diag, padding}), -1);

    logFile << "Shape of zero_padded_diag: " <<zero_padded_diag->get_shape().to_string() << std::endl;

    // 4.01 concatenate to additional padding of a shape [1, I, J, ..., M, N+k, N + 1 + k]

    auto zero_padded_diag_shape = make_shared<v3::ShapeOf>(zero_padded_diag);

    // Create shapes for additional rows and columns
    // For rows: Increase the second-to-last dimension by k
    // For columns: Increase the last dimension by k

    auto updated_shape = make_shared<v3::ScatterElementsUpdate>(
    zero_padded_diag_shape, // data input
    make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}), // indices
    make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}), // updates . **should be k**
    make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0}) // axis (only one in shape)
    );
    auto add_col_padding = make_shared<v3::Broadcast>(padding_value, updated_shape);

    logFile << "k: " <<k << std::endl;
    logFile << "Shape of add_col_padding: " <<add_col_padding->get_shape().to_string() << std::endl;

    auto zero_padded_diag_shape_2 = make_shared<v0::Concat>(OutputVector({zero_padded_diag, add_col_padding}), -1); 

    logFile << "Shape of zero_padded_diag_shape_2: " <<zero_padded_diag_shape_2->get_shape().to_string() << std::endl;

    // TODO: add concat logic here. add more to row in addition to column

    // reshape padded tensor to get a shape [I, J, ..., M, N * N + N]
    // 4.1 retrieve a part of the shape value [1, I, J, ..., M]
    auto new_shape_padded_diag1 =
        make_shared<v1::StridedSlice>(unsqueeze_diag_shape,
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-2}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
                                      std::vector<int64_t>({0}),
                                      std::vector<int64_t>({0}));
    logFile << "Shape of new_shape_padded_diag1: " <<new_shape_padded_diag1->get_shape().to_string() << std::endl;


    // 4.2 compute the last part of a shape that is [N * N + N]
    auto last_dim_squared = make_shared<v1::Multiply>(last_dim, last_dim);
    auto new_shape_padded_diag2 = make_shared<v1::Add>(last_dim_squared, last_dim);

    logFile << "Shape of new_shape_padded_diag2: " <<new_shape_padded_diag2->get_shape().to_string() << std::endl;

    // 4.3 compute a new shape and reshape padded diagonal
    auto new_shape_padded_diag =
        make_shared<v0::Concat>(OutputVector({new_shape_padded_diag1, new_shape_padded_diag2}), 0);
    auto reshaped_padded_diag = make_shared<v1::Reshape>(zero_padded_diag, new_shape_padded_diag, false);

    logFile << "Shape of new_shape_padded_diag " <<new_shape_padded_diag->get_shape().to_string() << std::endl;
    logFile << "Shape of reshaped_padded_diag: " <<reshaped_padded_diag->get_shape().to_string() << std::endl;

    // 5. cut off padding in the reshaped padded tensor to get a shape [1, I, J, ..., M, N * N]
    auto cut_padded_diag = make_shared<v8::Slice>(
        reshaped_padded_diag,
        make_shared<v0::Constant>(last_dim_squared->get_element_type(), Shape{1}, std::vector<int64_t>{0}),
        last_dim_squared,
        make_shared<v0::Constant>(last_dim_squared->get_element_type(), Shape{1}, std::vector<int64_t>{1}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    logFile << "Shape of cut_padded_diag: " <<cut_padded_diag->get_shape().to_string() << std::endl;

    // 6. return the expected shape for the result [I, J, ..., M, N, N]
    auto resulted_shape = make_shared<v0::Concat>(OutputVector({diag_shape, last_dim}), 0);
    auto resulted_diag = make_shared<v1::Reshape>(cut_padded_diag, resulted_shape, false);

    logFile << "Shape of  resulted_diag: " << resulted_diag->get_shape().to_string() << std::endl;

    set_node_name(node.get_name(), resulted_diag);

    logFile.close();
    return {resulted_diag};

}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
