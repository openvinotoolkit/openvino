// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique_consecutive.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_unique_consecutive(const NodeContext& context) {
    // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int dim=None) ->
    // (Tensor output, Tensor inverse_indices, Tensor counts)
    num_inputs_check(context, 1, 4);

    auto input = context.get_input(0);

    bool return_inverse = false;
    if (!context.input_is_none(1)) {
        return_inverse = context.const_input<bool>(1);
    }

    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_counts = context.const_input<bool>(2);
    }

    int64_t dim = -1;
    bool dim_is_none = context.input_is_none(3);
    if (!dim_is_none) {
        dim = context.const_input<int64_t>(3);
    }

    OutputVector outputs;
    Output<Node> prepared_input;
    Output<Node> axis_const;

    // Step 1: Choose the axis and prepare input
    if (dim_is_none) {
        // If dim is None, flatten the input tensor first
        auto shape = std::make_shared<v0::ShapeOf>(input);
        auto flatten_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
        prepared_input = context.mark_node(std::make_shared<v1::Reshape>(input, flatten_shape));
        // Use axis 0 for flattened tensor
        axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    } else {
        // Use input as-is with specified dimension
        prepared_input = input;
        // Handle negative axis value
        if (dim < 0) {
            auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(input));
            auto rank_scalar = context.mark_node(std::make_shared<v1::ReduceProd>(rank, v0::Constant::create(element::i64, Shape{}, {0})));
            auto dim_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
            axis_const = context.mark_node(std::make_shared<v1::Add>(dim_const, rank_scalar));
        } else {
            axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
        }
    }

    // Step 2: Compare the neighbors along axis a
    // build per-axis start/stop/step vectors (same length as input rank)
    auto shape = context.mark_node(std::make_shared<v0::ShapeOf>(prepared_input));
    auto zero_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));

    // broadcast scalars to shape-of-input to produce length-rank vectors
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(zero_scalar, shape));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one_scalar, shape));
    
    // head: start = [0, ...], stop = shape - 1 (exclusive)
    auto head_start = zeros;
    auto head_stop = context.mark_node(std::make_shared<v1::Subtract>(shape, ones));
    auto step = ones;

    // tail: start = [1, ...], stop = shape (exclusive -> equals shape)
    auto tail_start = ones;
    auto tail_stop = shape;

    // slice along all axes; only the axis of interest changes values in the vectors above
    auto head = context.mark_node(std::make_shared<v8::Slice>(prepared_input, head_start, head_stop, step));
    auto tail = context.mark_node(std::make_shared<v8::Slice>(prepared_input, tail_start, tail_stop, step));

    auto equal = context.mark_node(std::make_shared<v1::Equal>(head, tail));

    // Step 3 - Build a keep mask of run starts
    // change = not(equal) -> True where element i != i + 1 (i.e new run starts at i+1)
    auto change = context.mark_node(std::make_shared<v1::LogicalNot>(equal));

    // Prepend 'True' for the first element (first element always starts a run)
    // For the simple (flattened/1-D) case we can create a scalar/1-D true and concat
    auto true_one = context.mark_node(v0::Constant::create(element::boolean, Shape{1}, {true}));

    // axis index is known at conversion time in 'dim' (dim_is_none -> axis 0)
    int64_t axis_index = dim_is_none ? 0 : dim; // dim was read earlier from const input

    auto keep = context.mark_node(std::make_shared<v0::Concat>(OutputVector{true_one, change}, axis_index));

    // Step 4 - Get run start indices and the values output
    // NonZero(keep) -> indices tensor with shape [rank, N] (each column is a coordinate)
    auto nonzero = context.mark_node(std::make_shared<v3::NonZero>(keep));

    // Extract the row that corresponds to the slicing axis (row index == axis_index)
    auto axis_row_idx = context.mark_node(v0::Constant::create(element::i64, Shape{}, {axis_index}));
    auto nonzero_axis = context.mark_node(std::make_shared<v8::Gather>(nonzero, axis_row_idx, context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))));

    // nonzero_axis is a 1-D i64 tensor with the start indices along the chosen axis
    // Gather values from prepared_input along axis_const at positions nonzero_axis
    auto values = context.mark_node(std::make_shared<v8::Gather>(prepared_input, nonzero_axis, axis_const));

    // push the values (unique_consecutive output)
    outputs.push_back(values);

    // Step 5 - Compute counts (optional)
    // append sentinel = axis length to the list of start indices, then diff successive elements
    // axis_len_scalar = ShapeOf(prepared_input)[axis_const]
    auto concat_axis0 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));

    // get axis length (scalar)
    auto prepared_shape = context.mark_node(std::make_shared<v0::ShapeOf>(prepared_input)); // [rank]
    auto axis_len_scalar = context.mark_node(std::make_shared<v8::Gather>(prepared_shape, 
                                                                        context.mark_node(v0::Constant::create(element::i64, Shape{}, {axis_index})), 
                                                                        context.mark_node(v0::Constant::create(element::i64, Shape{}, {0})) ));

    // make axis_len 1-D so we can concat with nonzero axis
    auto axis_len_1d = context.mark_node(std::make_shared<v3::Unsqueeze>(axis_len_scalar, concat_axis0)); // shape{1}

    // concat starts + sentinel
    auto starts_with_sentinel = context.mark_node(std::make_shared<v0::Concat>(OutputVector{nonzero_axis, axis_len_1d}, 0)); // 1-D

    // compute size of concat (L = num_start + 1)
    auto starts_shape = context.mark_node(std::make_shared<v0::ShapeOf>(starts_with_sentinel)); // [1]
    auto size_scalar = context.mark_node(std::make_shared<v8::Gather>(starts_shape,
                                                                       context.mark_node(v0::Constant::create(element::i64, Shape{}, {0})),
                                                                       context.mark_node(v0::Constant::create(element::i64, Shape{}, {0})))); // scalar
    
    // build slice indices for head = starts_with_sentinel[0 : L-1] and tail = starts_with_sentinel[1 : L]
    auto zero_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto one_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));

    auto size_minus_one = context.mark_node(std::make_shared<v1::Subtract>(size_scalar, one_scalar)); // scalar

    auto head_stop_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(size_minus_one, concat_axis0)); // {L-1}
    auto tail_stop_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(size_scalar, concat_axis0));       // {L}

    auto head_indices = context.mark_node(std::make_shared<v8::Slice>(starts_with_sentinel, zero_1d, head_stop_1d, one_1d));
    auto tail_indices = context.mark_node(std::make_shared<v8::Slice>(starts_with_sentinel, one_1d, tail_stop_1d, one_1d));

    // counts = tail_indices - head_indices
    auto counts = context.mark_node(std::make_shared<v1::Subtract>(tail_indices, head_indices));

    // Step 6 - Compute inverse indices
    if (return_inverse) {
        // convert keep (bool) -> integer so we can CumSum
        auto keep_int = context.mark_node(std::make_shared<v1::Convert>(keep, element::i64));

        // CumSum along the chosen axis (inclusive). This produces per-position run labels:
        // e.g. keep = [1,0,1,0,...] -> sumsum = [1,1,2,2,...]
        auto cumsum = context.mark_node(std::make_shared<v3::CumSum>(keep_int, axis_const, /*exclusive*/ false, /*reverse*/ false));

        // Subtract 1 to make run ids 0-based: [1,1,2,2,...] -> [0,0,1,1,...]
        auto inverse_pre = context.mark_node(std::make_shared<v1::Subtract>(cumsum, one_scalar));

        // If we flattened the input (dim_is_none), reshape inverse back to original input shape
        Output<Node> inverse;
        if (dim_is_none) {
            auto orig_shape = context.mark_node(std::make_shared<v0::ShapeOf>(input)); // original input shape
            inverse = context.mark_node(std::make_shared<v1::Reshape>(inverse_pre, orig_shape));
        } else {
            // inverse already has same shape as prepared_input (same as input)
            inverse = inverse_pre;
        }

        // push inverse in the expected output order (values already pushed earlier)
        outputs.push_back(inverse);
    }

    if (return_counts) {
        outputs.push_back(counts);
    }

    return outputs;
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov