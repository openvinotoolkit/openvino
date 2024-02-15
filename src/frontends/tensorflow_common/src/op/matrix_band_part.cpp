// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/logical_and.hpp"
#include "openvino/op/less_equal.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_matrix_band_part_op(const NodeContext& node) {
    default_op_checks(node, 3, {"MatrixBandPart"});
    // Input tensor and parameters
    auto input = node.get_input(0);
    
    auto num_lower = node.get_input(1);
    auto num_upper = node.get_input(2);

    // Get the shape of the input tensor
    auto input_shape = make_shared<v3::ShapeOf>(input);

    // Find sizes m and n of the last two dimensions
    auto m = make_shared<v1::StridedSlice>(input_shape,
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {-2}),
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {-1}),
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {1}),
                                           std::vector<int64_t>({0}),
                                           std::vector<int64_t>({1}));

    auto n = make_shared<v1::StridedSlice>(input_shape,
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {-1}),
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {}),
                                           make_shared<v0::Constant>(element::i64, Shape{1}, {1}),
                                           std::vector<int64_t>({0}),
                                           std::vector<int64_t>({1}));

    // Generate ranges [0, m) and [0, n)
    auto range_m = make_shared<v1::Range>(m);
    auto range_n = make_shared<v1::Range>(n);

    // Unsqueeze ranges to have tensors of shapes [m, 1] and [1, n]
    auto unsqueeze_range_m = make_shared<v0::Unsqueeze>(range_m, make_shared<v0::Constant>(element::i64, Shape{1}, {1}));
    auto unsqueeze_range_n = make_shared<v0::Unsqueeze>(range_n, make_shared<v0::Constant>(element::i64, Shape{1}, {0}));

    // Create indicator function using logical operations
    // Create indicator function using logical operations
    auto in_band_rows = make_shared<v1::LogicalAnd>(
        make_shared<v1::LogicalOr>(
            make_shared<v1::Less>(make_shared<v1::Constant>(element::i64, Shape{}, {num_lower}), make_shared<v1::Constant>(element::i64, Shape{}, {0})),
            make_shared<v1::LessEqual>(make_shared<v1::Subtract>(m, n), make_shared<v1::Constant>(element::i64, Shape{}, {num_lower}))
        ),
        make_shared<v1::LogicalOr>(
            make_shared<v1::Less>(make_shared<v1::Constant>(element::i64, Shape{}, {num_upper}), make_shared<v1::Constant>(element::i64, Shape{}, {0})),
            make_shared<v1::LessEqual>(make_shared<v1::Subtract>(n, m), make_shared<v1::Constant>(element::i64, Shape{}, {num_upper}))
        )
    );

    auto in_band_cols = make_shared<v1::LogicalAnd>(
        make_shared<v1::LogicalOr>(
            make_shared<v1::Less>(make_shared<v1::Constant>(element::i64, Shape{}, {num_lower}), make_shared<v1::Constant>(element::i64, Shape{}, {0})),
            make_shared<v1::LessEqual>(make_shared<v1::Subtract>(m, n), make_shared<v1::Constant>(element::i64, Shape{}, {num_lower}))
        ),
        make_shared<v1::LogicalOr>(
            make_shared<v1::Less>(make_shared<v1::Constant>(element::i64, Shape{}, {num_upper}), make_shared<v1::Constant>(element::i64, Shape{}, {0})),
            make_shared<v1::LessEqual>(make_shared<v1::Subtract>(n, m), make_shared<v1::Constant>(element::i64, Shape{}, {num_upper}))
        )
    );

    auto in_band_indicator = make_shared<v1::LogicalAnd>(in_band_rows, in_band_cols);

    // Unsqueeze the indicator function to have a tensor of shape [1,..,1, m, n]
    auto unsqueeze_indicator = make_shared<v0::Unsqueeze>(in_band_indicator, make_shared<v0::Constant>(element::i64, Shape{}, {1}));

    // Extract or modify the elements based on the indicator function
    auto result = make_shared<v3::Where>(unsqueeze_indicator, input, make_shared<v0::Constant>(input->get_element_type(), Shape{}, {0}));

    // Return the result
    set_node_name(node.get_name(), result);
    return {result};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
