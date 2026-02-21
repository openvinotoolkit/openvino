
#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

// Optimized MatrixDiagV3 implementation for k = 0 (main diagonal only)
OutputVector translate_matrix_diag_v3_op(const NodeContext& node) {
    default_op_checks(node, 5, {"MatrixDiagV3"});

    auto diagonal = node.get_input(0);
    auto k = node.get_input(1);
    auto num_rows = node.get_input(2);
    auto num_cols = node.get_input(3);
    auto padding_value = node.get_input(4);

    auto align = node.get_attribute<string>("align", "RIGHT_LEFT");

    auto diagonal_shape_node = make_shared<v3::ShapeOf>(diagonal, element::i64);

    auto k_shape = k.get_partial_shape();
    Output<Node> k_lower, k_upper;

    if (k_shape.is_static() && shape_size(k_shape.to_shape()) == 1) {
        k_lower = make_shared<v0::Squeeze>(k);
        k_upper = k_lower;
    } else {
        auto axis = make_shared<v0::Constant>(element::i64, Shape{}, 0);
        k_lower = make_shared<v0::Squeeze>(
            make_shared<v8::Gather>(k, make_shared<v0::Constant>(element::i64, Shape{}, 0), axis));
        k_upper = make_shared<v0::Squeeze>(
            make_shared<v8::Gather>(k, make_shared<v0::Constant>(element::i64, Shape{}, 1), axis));
    }

    auto zero_i32 = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto k_lower_i32 = make_shared<v0::Convert>(k_lower, element::i32);
    auto k_upper_i32 = make_shared<v0::Convert>(k_upper, element::i32);

    auto k_lower_input = k_lower_i32->input_value(0);
    auto k_upper_input = k_upper_i32->input_value(0);

    auto k_lower_const = ov::as_type_ptr<v0::Constant>(k_lower_input.get_node_shared_ptr());
    auto k_upper_const = ov::as_type_ptr<v0::Constant>(k_upper_input.get_node_shared_ptr());

    OPENVINO_ASSERT(k_lower_const && k_upper_const,
                    "MatrixDiagV3: only constant k is supported for k=0 implementation");

    OPENVINO_ASSERT(k_lower_const->cast_vector<int32_t>()[0] == 0 && k_upper_const->cast_vector<int32_t>()[0] == 0,
                    "MatrixDiagV3: only k=0 is supported in this implementation");

    // ===== K=0 OPTIMIZED PATH =====

    auto unsqueeze_axis = make_shared<v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, -1});
    auto unsqueeze_diag = make_shared<v0::Unsqueeze>(diagonal, unsqueeze_axis);
    auto unsqueeze_diag_shape = make_shared<v3::ShapeOf>(unsqueeze_diag, element::i64);

    auto last_dim =
        make_shared<v1::StridedSlice>(unsqueeze_diag_shape,
                                      make_shared<v0::Constant>(element::i64, Shape{1}, vector<int64_t>{-2}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, vector<int64_t>{-1}),
                                      make_shared<v0::Constant>(element::i64, Shape{1}, vector<int64_t>{1}),
                                      vector<int64_t>{0},
                                      vector<int64_t>{0});

    auto neg_one_i32 = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto last_dim_squeezed = make_shared<v0::Squeeze>(last_dim);
    auto last_dim_i32 = make_shared<v0::Convert>(last_dim_squeezed, element::i32);

    auto num_rows_scalar = make_shared<v0::Squeeze>(num_rows);
    auto num_cols_scalar = make_shared<v0::Squeeze>(num_cols);

    auto rows_is_auto = make_shared<v1::Equal>(num_rows_scalar, neg_one_i32);
    auto cols_is_auto = make_shared<v1::Equal>(num_cols_scalar, neg_one_i32);

    auto output_rows = make_shared<v1::Select>(rows_is_auto, last_dim_i32, num_rows_scalar);
    auto output_cols = make_shared<v1::Select>(cols_is_auto, last_dim_i32, num_cols_scalar);

    auto output_rows_i64 = make_shared<v0::Convert>(output_rows, element::i64);
    auto output_cols_i64 = make_shared<v0::Convert>(output_cols, element::i64);

    auto diag_shape = make_shared<v3::ShapeOf>(diagonal, element::i64);

    auto one_i64 = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto output_rows_unsqueezed =
        make_shared<v0::Unsqueeze>(output_rows_i64, make_shared<v0::Constant>(element::i64, Shape{1}, 0));
    auto padding_shape = make_shared<v0::Concat>(OutputVector{one_i64, diag_shape, output_rows_unsqueezed}, 0);

    auto padding_val_scalar = make_shared<v0::Squeeze>(padding_value);
    auto padding = make_shared<v3::Broadcast>(padding_val_scalar, padding_shape);

    auto zero_padded = make_shared<v0::Concat>(OutputVector{unsqueeze_diag, padding}, -1);
    auto batch_shape = make_shared<v1::StridedSlice>(unsqueeze_diag_shape,
                                                     make_shared<v0::Constant>(element::i64, Shape{1}, 0),
                                                     make_shared<v0::Constant>(element::i64, Shape{1}, -2),
                                                     make_shared<v0::Constant>(element::i64, Shape{1}, 1),
                                                     vector<int64_t>{0},
                                                     vector<int64_t>{0});

    auto n_plus_rows = make_shared<v1::Add>(
        make_shared<v0::Unsqueeze>(last_dim_squeezed, make_shared<v0::Constant>(element::i64, Shape{1}, 0)),
        output_rows_unsqueezed);
    auto flat_size = make_shared<v1::Multiply>(
        make_shared<v0::Unsqueeze>(last_dim_squeezed, make_shared<v0::Constant>(element::i64, Shape{1}, 0)),
        n_plus_rows);

    auto reshape_shape1 = make_shared<v0::Concat>(OutputVector{batch_shape, flat_size}, 0);
    auto reshaped = make_shared<v1::Reshape>(zero_padded, reshape_shape1, false);

    auto output_size = make_shared<v1::Multiply>(
        output_rows_unsqueezed,
        make_shared<v0::Unsqueeze>(output_cols_i64, make_shared<v0::Constant>(element::i64, Shape{1}, 0)));
    auto sliced = make_shared<v8::Slice>(reshaped,
                                         make_shared<v0::Constant>(element::i64, Shape{1}, 0),
                                         output_size,
                                         make_shared<v0::Constant>(element::i64, Shape{1}, 1),
                                         make_shared<v0::Constant>(element::i64, Shape{1}, -1));

    auto output_cols_unsqueezed =
        make_shared<v0::Unsqueeze>(output_cols_i64, make_shared<v0::Constant>(element::i64, Shape{1}, 0));
    auto final_shape =
        make_shared<v0::Concat>(OutputVector{diag_shape, output_rows_unsqueezed, output_cols_unsqueezed}, 0);
    auto result = make_shared<v1::Reshape>(sliced, final_shape, false);

    set_node_name(node.get_name(), result);
    return {result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
