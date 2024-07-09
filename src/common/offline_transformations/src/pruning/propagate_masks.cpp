// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iterator>
#include <memory>

#include "mask_attribute.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"
#include "openvino/util/log.hpp"
#include "pruning.hpp"

namespace ov {
namespace pass {
namespace mask_propagation {

class MatMul;
class Convolution;
class GroupConvolution;
class GroupConvolutionReshape;
class Elementwise;
class PassThrough;
class Reduce;
class Reshape;
class Transpose;
class StopPropagation;
class SkipPropagation;
class FakeQuantize;
class Concat;

}  // namespace mask_propagation
}  // namespace pass
}  // namespace ov

namespace ov {
namespace pass {
namespace mask_propagation {

class VariadicSplit;
class Split;

}  // namespace mask_propagation
}  // namespace pass
}  // namespace ov

static ov::Shape broadcast_shape_to_rank(ov::Shape shape_to_broadcast, int64_t dst_rank) {
    auto initial_rank = static_cast<int64_t>(shape_to_broadcast.size());
    auto num_of_broadcased_dims = dst_rank - initial_rank;
    std::vector<size_t> dims(num_of_broadcased_dims, 1);
    dims.insert(dims.end(), shape_to_broadcast.begin(), shape_to_broadcast.end());
    auto new_shape = ov::Shape(dims);
    return new_shape;
}

class ov::pass::mask_propagation::MatMul : public MatcherPass {
public:
    MatMul() {
        auto a = pattern::any_input(pattern::has_static_shape());
        auto b = pattern::any_input(pattern::has_static_shape());
        auto matmul = pattern::wrap_type<opset10::MatMul>({a, b});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_a = pattern_map.at(a);
            const auto& m_b = pattern_map.at(b);
            const auto& m_matmul = pattern_map.at(matmul);

            auto a_mask = getMask(m_a);
            auto b_mask = getMask(m_b);

            if (!a_mask && !b_mask) {
                OPENVINO_DEBUG("No mask for any input of ", m_matmul.get_node()->get_friendly_name(), "\n");
                return false;
            }
            if (!b_mask) {
                OPENVINO_DEBUG("No mask for input b of ", m_matmul.get_node()->get_friendly_name(), "\n");
                return false;
            }

            const auto matmul_range = m_matmul.get_shape().size();
            if (matmul_range < 2) {
                OPENVINO_DEBUG("Matmul operation with rank = 1 is not supported by pruning algo by now\n");
                return false;
            }

            ov::Mask* a_mask_row = nullptr;
            if (a_mask)
                a_mask_row = a_mask.get();
            auto b_mask_row = b_mask.get();

            const auto matmul_op = std::dynamic_pointer_cast<opset10::MatMul>(m_matmul.get_node_shared_ptr());
            const auto transpose_a = matmul_op->get_transpose_a();
            const auto transpose_b = matmul_op->get_transpose_b();

            const auto shape_a = m_a.get_shape();
            const auto shape_b = m_b.get_shape();

            const auto a_inner_dim = (transpose_a) ? shape_a.size() - 2 : shape_a.size() - 1;
            const auto a_outer_dim = (transpose_a) ? shape_a.size() - 1 : shape_a.size() - 2;
            const auto b_inner_dim = (transpose_b) ? shape_b.size() - 1 : shape_b.size() - 2;
            const auto b_outer_dim = (transpose_b) ? shape_b.size() - 2 : shape_b.size() - 1;

            auto matmul_mask = std::make_shared<ov::Mask>(matmul_range);
            auto matmul_mask_row = matmul_mask.get();
            const auto matmul_cols_dim = matmul_range - 1;
            const auto matmul_rows_dim = matmul_range - 2;

            // Connect a with matmul mask
            if (a_mask) {
                bool init = true;
                a_mask->add_callback(
                    [=](ov::Mask::Ptr cur_mask) -> bool {
                        auto result_mask = std::make_shared<ov::Mask>(cur_mask->size());
                        result_mask->copy_value_from_mask_reversed(matmul_mask_row);
                        result_mask->at(a_inner_dim) = b_mask_row->at(b_inner_dim);
                        result_mask->at(a_outer_dim) = matmul_mask_row->at(matmul_rows_dim);
                        cur_mask->copy_value_from_mask(result_mask.get());
                        return true;
                    },
                    matmul_mask);
                matmul_mask->add_callback(
                    [=](ov::Mask::Ptr cur_mask) mutable -> bool {
                        auto result_mask = std::make_shared<ov::Mask>(cur_mask->size());
                        result_mask->copy_value_from_mask(cur_mask.get());
                        result_mask->copy_value_from_mask_reversed(a_mask_row);
                        if (init) {
                            result_mask->at(matmul_cols_dim) = b_mask_row->at(b_outer_dim);
                            init = false;
                        } else {
                            result_mask->at(matmul_cols_dim) = cur_mask->at(matmul_cols_dim);
                        }
                        result_mask->at(matmul_rows_dim) = a_mask_row->at(a_outer_dim);
                        if (a_mask_row->at(a_inner_dim) != b_mask_row->at(b_inner_dim))
                            cur_mask->initialize_dependencies();
                        cur_mask->copy_value_from_mask_reversed(result_mask.get());
                        return true;
                    },
                    a_mask);
            }
            // connect b with matmul mask
            b_mask->add_callback(
                [=](ov::Mask::Ptr cur_mask) -> bool {
                    auto result_mask = std::make_shared<ov::Mask>(cur_mask->size());
                    result_mask->copy_value_from_mask_reversed(matmul_mask_row);
                    if (a_mask_row)
                        result_mask->at(b_inner_dim) = a_mask_row->at(a_inner_dim);
                    else
                        result_mask->at(b_inner_dim).clear();
                    result_mask->at(b_outer_dim) = matmul_mask_row->at(matmul_cols_dim);  // TODO: remove this line
                    cur_mask->copy_value_from_mask(result_mask.get());
                    return true;
                },
                matmul_mask);
            matmul_mask->add_callback(
                [=](ov::Mask::Ptr cur_mask) -> bool {
                    if (a_mask_row) {
                        auto result_mask = std::make_shared<ov::Mask>(cur_mask->size());
                        result_mask->copy_value_from_mask(cur_mask.get());
                        result_mask->copy_value_from_mask_reversed(b_mask_row);
                        result_mask->at(matmul_rows_dim) = cur_mask->at(matmul_rows_dim);
                        result_mask->at(matmul_cols_dim) = b_mask_row->at(b_outer_dim);
                        if (a_mask_row->at(a_inner_dim) != b_mask_row->at(b_inner_dim))
                            cur_mask->initialize_dependencies();
                        cur_mask->copy_value_from_mask(result_mask.get());
                    } else {
                        cur_mask->clean_dim_values();
                        cur_mask->at(matmul_cols_dim) = b_mask_row->at(b_outer_dim);
                    }
                    return true;
                },
                b_mask);

            bool status;
            if (!a_mask || a_mask->all_dims_are_empty())
                status = matmul_mask->apply_callback(b_mask);
            else
                status = matmul_mask->apply_callback(a_mask);

            if (!status)
                return false;

            setMask(m_matmul, matmul_mask);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul, "MatMulMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Convolution : public MatcherPass {
public:
    Convolution() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input(pattern::has_static_shape());
        auto conv = pattern::wrap_type<opset10::Convolution>({input, weights});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_weights = pattern_map.at(weights);
            const auto& m_output = pattern_map.at(conv);
            const auto& m_input = pattern_map.at(input);

            auto weights_mask = getMask(m_weights);

            // Nullptr in weights-mask means that mask for this node wasn't initialized earlier.
            // Weights mask for convolution should be initialized in the InitMasks pass (and propagate after it).
            // If mask isn't initialized - this weights (and hence all convolution) can't be pruned for some reason.
            if (!weights_mask) {
                OPENVINO_DEBUG("No weights mask for ", m_output.get_node()->get_friendly_name(), "\n");
                return false;
            }
            auto weights_mask_row = weights_mask.get();

            // Create output mask that describes which channel dimensions will be removed
            auto conv_mask = std::make_shared<ov::Mask>(m_weights.get_shape().size());
            auto conv_mask_row = conv_mask.get();
            auto input_mask = getMask(m_input);
            ov::Mask* input_mask_row = nullptr;
            if (input_mask)
                input_mask_row = input_mask.get();

            const auto conv_mask_callback = [input_mask_row, weights_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1 /*input data channel*/) = weights_mask_row->at(0 /* weights output channel dim*/);
                if (input_mask_row && input_mask_row->at(1) != weights_mask_row->at(1))
                    cur_mask->initialize_dependencies();
                return true;
            };

            if (input_mask) {
                // Weights input channel is connected to the convolution input channel dimension
                // so we update weights mask to be aligned with input shape.
                conv_mask->add_callback(conv_mask_callback, input_mask);
                input_mask->add_callback(
                    [weights_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                        cur_mask->at(1) = weights_mask_row->at(1);
                        return true;
                    },
                    conv_mask);
            }

            conv_mask->add_callback(conv_mask_callback, weights_mask);
            weights_mask->add_callback(
                [input_mask_row, conv_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(0) = conv_mask_row->at(1);
                    if (input_mask_row)
                        cur_mask->at(1) = input_mask_row->at(1);
                    return true;
                },
                conv_mask);

            bool status;
            if (input_mask)
                status = conv_mask->apply_callback(input_mask);
            else
                status = conv_mask->apply_callback(weights_mask);

            if (!status)
                return false;

            setMask(m_output, conv_mask);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, "ConvolutionMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::GroupConvolution : public MatcherPass {
public:
    GroupConvolution() {
        auto input = pattern::any_input(pattern::has_static_dim(1));
        auto weights = pattern::any_input(pattern::has_static_shape());
        auto group_conv = pattern::wrap_type<opset10::GroupConvolution>({input, weights});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_weights = pattern_map.at(weights);
            const auto& m_output = pattern_map.at(group_conv);
            const auto& m_input = pattern_map.at(input);

            // TODO: check static rank in pattern, use only particular dims
            auto weights_shape = m_weights.get_shape();
            auto input_shape = m_input.get_partial_shape();
            // support only depthwise convolutions
            if (weights_shape[0] != static_cast<size_t>(input_shape[1].get_length())) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask)
                return false;
            auto input_mask_row = input_mask.get();

            auto weights_mask = getMask(m_weights);
            if (!weights_mask) {
                // Setting mask only if weights are constant
                if (ov::is_type<opset10::Constant>(m_output.get_node_shared_ptr())) {
                    weights_mask = std::make_shared<ov::Mask>(weights_shape.size());
                    setMask(m_weights, weights_mask);
                } else {
                    OPENVINO_DEBUG("GroupConvolution: No weights mask and weights aren't constant for ",
                                   *m_output.get_node(),
                                   "\n");
                    return false;
                }
            }
            auto weights_mask_row = weights_mask.get();

            auto conv_mask = std::make_shared<ov::Mask>(input_shape.rank().get_length());
            auto conv_mask_row = conv_mask.get();

            conv_mask->add_callback(
                [input_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1 /*input data channel*/) = input_mask_row->at(1 /*output data channel*/);
                    return true;
                },
                input_mask);

            input_mask->add_callback(
                [conv_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1 /*output data channel*/) = conv_mask_row->at(1 /*input data channel*/);
                    return true;
                },
                conv_mask);

            conv_mask->add_callback(
                [weights_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(1 /*input data channel*/) = weights_mask_row->at(0 /*weights output channel dim*/);
                    return true;
                },
                weights_mask);

            weights_mask->add_callback(
                [conv_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(0 /*weights output channel dim*/) = conv_mask_row->at(1 /*output data channel*/);
                    return true;
                },
                conv_mask);

            if (!conv_mask->apply_callback(input_mask)) {
                return false;
            }

            setMask(m_output, conv_mask);

            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv, "GroupConvolutionMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::GroupConvolutionReshape : public MatcherPass {
public:
    GroupConvolutionReshape() {
        auto input = pattern::any_input(pattern::has_static_shape());
        auto shape = pattern::any_input();
        // Working only for Reshapes on Group Convolution weights
        auto reshape = pattern::wrap_type<opset10::Reshape>({input, shape}, pattern::consumers_count(1));
        auto gconv =
            pattern::wrap_type<opset10::GroupConvolution>({pattern::any_input(), reshape}, pattern::has_static_shape());

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_shape = pattern_map.at(shape);
            const auto& m_output = pattern_map.at(reshape);
            const auto& m_input = pattern_map.at(input);

            auto shape_val = m_shape.get_node_shared_ptr();

            // In Depthwise Convolutions Reshape on weights just add additional dimension for output channels count
            // (1 in case of the depthwise) of kernel.
            // Example: Reshape from [G, 1 (I), X, Y, Z] -> [G, 1 (O), 1 (I), X, Y, Z], where G - group numbers,
            // X, Y, Z -  spartial dimensions (can be only X or X, Y), I, O - number of input/output channels of kernel.

            // Checking that matched Reshape meets this conditions (add 1-d dim on 1 position of shape constant)
            auto inp_shape = m_input.get_shape();
            auto out_shape = m_output.get_shape();
            inp_shape.insert(inp_shape.begin() + 1, 1);
            if (inp_shape != out_shape || out_shape.size() != 5) {
                return false;
            }

            auto input_mask = getMask(m_input);
            if (!input_mask) {
                return false;
            }

            const auto constant = ov::util::get_constant_from_source(m_shape.get_node_shared_ptr());
            if (!constant) {
                OPENVINO_DEBUG("Can't get constant from source node ", m_shape.get_node()->get_friendly_name());
                return false;
            }
            auto input_mask_row = input_mask.get();
            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());

            auto output_mask_row = output_mask.get();

            // Depthwise Convolution pruned only by input channels (== groups) ->
            // Propagating mask from Group (0) dim in Reshape input to Group (0) dim in Reshape output and back
            input_mask->add_callback(
                [output_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(0) = output_mask_row->at(0);
                    return true;
                },
                output_mask);
            output_mask->add_callback(
                [input_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->at(0) = input_mask_row->at(0);
                    return true;
                },
                input_mask);
            output_mask->apply_callback(input_mask);

            setMask(m_output, output_mask);
            // To allow pruning on weights (allow reshape input Group (0) dim changing) modify Reshape Shape input:
            // [G, 1, 1, X, Y, Z] by [-1, 1, 1, X, Y, Z].

            const auto m_shape_consumers = m_shape.get_target_inputs();
            const auto output_shape = constant->get_shape();
            const auto axis = opset10::Constant::create(element::i8, {}, {0});
            auto dims_to_keep_vec = std::vector<size_t>{2, 3, 4};

            const auto dims_to_keep =
                opset10::Constant::create(m_shape.get_element_type(), {dims_to_keep_vec.size()}, dims_to_keep_vec);
            const auto gather = std::make_shared<opset10::Gather>(m_shape, dims_to_keep, axis);
            const auto concat = std::make_shared<opset10::Concat>(
                NodeVector{opset10::Constant::create(m_shape.get_element_type(), {2}, {-1, 1}), gather},
                0);
            for (auto& consumer : m_shape_consumers)
                consumer.replace_source_output(concat);

            // This transformation propagates only Reshape mask and doesn't do anything with GroupConvolution.
            // So, not to disable GroupConvolution mask propagation we return false here.
            return false;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(gconv, "GroupConvolutionReshapeMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Elementwise : public MatcherPass {
public:
    Elementwise() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto eltwise =
            pattern::wrap_type<opset10::Add, opset10::Subtract, opset10::Maximum, opset10::Minimum, opset10::Multiply>(
                {input, weights},
                pattern::has_static_rank());
        // TODO: add Div, Power support

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_output = pattern_map.at(eltwise);
            // Inputs are taken in deterministic way
            const auto& m_input = m_output.get_node_shared_ptr()->get_input_source_output(0);
            const auto& m_weights = m_output.get_node_shared_ptr()->get_input_source_output(1);

            const auto& input_rank = m_input.get_partial_shape().rank().get_length();
            const auto& weights_rank = m_weights.get_partial_shape().rank().get_length();
            // Here assuming that masks can be propagated only through 3/4 dimensional tensors
            // (since channel dim is necessary) or tensors with equal rank.
            if (!((weights_rank > 2 && input_rank > 2) || weights_rank == input_rank))
                return false;

            if (m_output.get_node_shared_ptr()->get_autob() != op::AutoBroadcastType::NUMPY) {
                OPENVINO_DEBUG("Can't propagate mask through ",
                               m_output.get_node()->get_friendly_name(),
                               " because node is using unsupported broadcast mode.\n");
                return false;
            }
            // Case when input masks should be united instead of intersection
            bool union_eltwise_type = ov::is_type<opset10::Multiply>(m_output.get_node_shared_ptr());

            using dims_set = std::set<int64_t>;
            auto input_shape_broadcasted_dims = dims_set();
            auto weights_shape_broadcasted_dims = dims_set();
            auto input_shape_mask = dims_set();
            auto weights_shape_mask = dims_set();
            auto input_shape = ov::Shape();
            auto weights_shape = ov::Shape();
            if (m_input.get_partial_shape().is_static() && m_weights.get_partial_shape().is_static()) {
                // Compute brodcasted dims
                input_shape = m_input.get_shape();
                weights_shape = m_weights.get_shape();
                const int64_t input_shape_size_diff =
                    static_cast<int64_t>(input_shape.size()) - static_cast<int64_t>(weights_shape.size());
                const int64_t weights_shape_size_diff = -input_shape_size_diff;
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    const int64_t shifted_elem = i + weights_shape_size_diff;
                    if (shifted_elem >= 0 && input_shape[i] == 1 && weights_shape[shifted_elem] != 1)
                        input_shape_broadcasted_dims.insert(i);
                    if (shifted_elem < 0 && input_shape[i] != 1)
                        weights_shape_broadcasted_dims.insert(shifted_elem);
                }
                for (size_t i = 0; i < weights_shape.size(); ++i) {
                    const int64_t shifted_elem = i + input_shape_size_diff;
                    if (shifted_elem >= 0 && weights_shape[i] == 1 && input_shape[shifted_elem] != 1)
                        weights_shape_broadcasted_dims.insert(i);
                    if (shifted_elem < 0 && weights_shape[i] != 1)
                        input_shape_broadcasted_dims.insert(shifted_elem);
                }
                const auto ge_zero_pred = [](int64_t x) {
                    return x >= 0;
                };
                std::copy_if(input_shape_broadcasted_dims.begin(),
                             input_shape_broadcasted_dims.end(),
                             std::inserter(input_shape_mask, input_shape_mask.begin()),
                             ge_zero_pred);
                std::copy_if(weights_shape_broadcasted_dims.begin(),
                             weights_shape_broadcasted_dims.end(),
                             std::inserter(weights_shape_mask, weights_shape_mask.begin()),
                             ge_zero_pred);

                for (const auto elem : weights_shape_broadcasted_dims) {
                    const auto shifted_elem = elem + input_shape_size_diff;
                    if (shifted_elem >= 0)
                        input_shape_mask.insert(shifted_elem);
                }

                for (const auto elem : input_shape_broadcasted_dims) {
                    const auto shifted_elem = elem + weights_shape_size_diff;
                    if (shifted_elem >= 0)
                        weights_shape_mask.insert(shifted_elem);
                }
            }

            // Prevent case when input_shape and weights_shape both has broadcasted dims
            if (input_shape_broadcasted_dims.size() && weights_shape_broadcasted_dims.size()) {
                OPENVINO_DEBUG("Can't propagate mask through ",
                               m_output.get_node()->get_friendly_name(),
                               " because both input shapes contains broadcasted dims.\n");
                return false;
            }

            const auto input_axis_set = get_axis_set(input_shape);
            InitConstMask(input_axis_set).apply(m_input.get_node_shared_ptr());
            auto input_mask = getMask(m_input);

            const auto weights_axis_set = get_axis_set(weights_shape);
            InitConstMask(weights_axis_set).apply(m_weights.get_node_shared_ptr());
            auto weights_mask = getMask(m_weights);

            if (input_shape_broadcasted_dims.size()) {
                // Swap input and weights inputs
                std::swap(input_mask, weights_mask);
                std::swap(input_shape_mask, weights_shape_mask);
                std::swap(input_shape_broadcasted_dims, weights_shape_broadcasted_dims);
            }

            if (!input_mask) {
                OPENVINO_DEBUG("No input mask for: ", m_output.get_node()->get_friendly_name(), "\n");
                return false;
            }
            if (!weights_mask) {
                // Set dummy mask to weight input in case this input has no mask
                // and has broadcastable dimentions
                if (!weights_shape_broadcasted_dims.size()) {
                    OPENVINO_DEBUG("No weights mask for: ", m_output.get_node()->get_friendly_name(), "\n");
                    return false;
                }
                weights_mask = std::make_shared<ov::Mask>(m_weights.get_partial_shape().rank().get_length());
                setMask(m_weights, weights_mask);
            }

            auto input_mask_row = input_mask.get();
            auto weights_mask_row = weights_mask.get();
            // Merging masks from two inputs
            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            auto out_mask_callback = [=](ov::Mask::Ptr cur_mask) -> bool {
                ov::Mask::Ptr result_mask;
                if (union_eltwise_type) {
                    result_mask = input_mask_row->union_masks_reversed(weights_mask_row);
                } else {
                    result_mask = input_mask_row->intersect_masks_reversed(weights_mask_row);
                }
                cur_mask->copy_value_from_mask_reversed(result_mask.get());
                cur_mask->copy_value_from_mask_reversed_masked(input_mask_row, input_shape_mask, true);
                return true;
            };
            output_mask->add_callback(out_mask_callback, input_mask);

            input_mask->add_callback(
                [weights_mask_row, input_shape_mask](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->copy_value_from_mask_reversed_masked(weights_mask_row, input_shape_mask);
                    return true;
                },
                weights_mask);
            input_mask->add_callback(
                [output_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->copy_value_from_mask_reversed(output_mask_row);
                    return true;
                },
                output_mask);
            weights_mask->add_callback(
                [input_mask_row, weights_shape_mask](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->copy_value_from_mask_reversed_masked(input_mask_row, weights_shape_mask);
                    for (const auto dim : weights_shape_mask)
                        cur_mask->at(dim).clear();
                    return true;
                },
                input_mask);

            output_mask->apply_callback(input_mask);
            weights_mask->apply_callback(input_mask);

            setMask(m_output, output_mask);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(eltwise, "ElementwiseMaskPropagation");
        register_matcher(m, callback);
    }

private:
    static ov::AxisSet get_axis_set(const ov::Shape shape) {
        ov::AxisSet ret_val;
        if (shape.size()) {
            ret_val = ov::AxisSet();
            for (size_t i = 0; i < shape.size(); ++i)
                ret_val.insert(i);
        } else {
            ret_val = ov::AxisSet({0, 1, 2, 3});
        }
        return ret_val;
    }
};

class ov::pass::mask_propagation::FakeQuantize : public MatcherPass {
public:
    FakeQuantize() {
        auto input = pattern::any_input(pattern::has_static_shape());
        auto input_low = pattern::any_input(pattern::has_static_shape());
        auto input_high = pattern::any_input(pattern::has_static_shape());
        auto output_low = pattern::any_input(pattern::has_static_shape());
        auto output_high = pattern::any_input(pattern::has_static_shape());
        auto fake_quantize =
            pattern::wrap_type<opset10::FakeQuantize>({input, input_low, input_high, output_low, output_high});
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_input = pattern_map.at(input);
            const auto& m_input_low = pattern_map.at(input_low);
            const auto& m_input_high = pattern_map.at(input_high);
            const auto& m_output_low = pattern_map.at(output_low);
            const auto& m_output_high = pattern_map.at(output_high);
            const auto& m_output = pattern_map.at(fake_quantize);

            auto input_mask = getMask(m_input);

            // Input mask is the only source of pruning in FQ
            if (!input_mask) {
                OPENVINO_DEBUG("FakeQuantize: No input mask for ", *m_output.get_node(), "\n");
                return false;
            }

            auto input_mask_row = input_mask.get();

            // Propagate input mask to output mask and in the opposite direction
            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            // Output mask is equal to input mask
            auto output_mask_callback = [input_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask(input_mask_row);
                return true;
            };

            auto input_mask_callback = [output_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                cur_mask->copy_value_from_mask(output_mask_row);
                return true;
            };

            output_mask->add_callback(output_mask_callback, input_mask);
            input_mask->add_callback(input_mask_callback, output_mask);

            // Calculate output mask
            output_mask->apply_callback(input_mask);
            setMask(m_output, output_mask);

            auto input_low_size = shape_size(m_input_low.get_shape());
            auto input_high_size = shape_size(m_input_high.get_shape());
            auto output_low_size = shape_size(m_output_low.get_shape());
            auto output_high_size = shape_size(m_output_high.get_shape());

            // In the per-tensor case FQ params shouldn't be pruned
            if (input_low_size == 1 && output_low_size == 1 && input_high_size == 1 && output_high_size == 1) {
                return true;
            }

            // If input/output ranges in FQ should be broadcasted to input shape -> broadcast this consant values
            // for the convenience of working with the masks
            NodeVector fq_params_nodes{m_input_low.get_node_shared_ptr(),
                                       m_input_high.get_node_shared_ptr(),
                                       m_output_low.get_node_shared_ptr(),
                                       m_output_high.get_node_shared_ptr()};
            auto fq_node = std::dynamic_pointer_cast<opset10::FakeQuantize>(m_output.get_node_shared_ptr());
            if (!fq_node)
                return false;
            size_t idx = 0;
            if (fq_node->get_auto_broadcast() != ov::op::AutoBroadcastType::NONE) {
                for (const auto& node : fq_params_nodes) {
                    auto const_node = std::dynamic_pointer_cast<op::v0::Constant>(node);
                    if (!const_node)
                        OPENVINO_THROW("Unexpected operation type.");
                    auto new_shape = broadcast_shape_to_rank(const_node->get_shape(),
                                                             m_input.get_partial_shape().rank().get_length());
                    auto new_const = std::make_shared<op::v0::Constant>(*const_node, new_shape);
                    new_const->set_friendly_name(const_node->get_friendly_name());
                    ov::copy_runtime_info(const_node, new_const);
                    ov::replace_node(const_node, new_const);
                    fq_params_nodes[idx++] = new_const;
                }
            }

            auto fq_params_mask_callback = [input_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                cur_mask->at(1 /* fq params have same shapes as input */) =
                    input_mask_row->at(1 /* channel dim in data */);
                return true;
            };

            for (const auto& fq_param : fq_params_nodes) {
                auto mask = std::make_shared<ov::Mask>(fq_param->get_shape().size());
                mask->add_callback(fq_params_mask_callback, input_mask);
                input_mask->add_callback(
                    [mask](ov::Mask::Ptr cur_mask) -> bool {
                        return true;
                    },
                    mask);
                mask->apply_callback(input_mask);
                setMask(fq_param->output(0), mask);
            }

            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(fake_quantize, "FakeQuantizeMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Concat : public MatcherPass {
public:
    Concat() {
        auto concat = pattern::wrap_type<opset10::Concat>(pattern::has_static_shape());

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_output = pattern_map.at(concat);
            auto concat_ptr = std::dynamic_pointer_cast<opset10::Concat>(m_output.get_node_shared_ptr());
            if (!concat_ptr) {
                return false;
            }
            int64_t axis = -1;
            if (concat_ptr->get_output_partial_shape(0).rank().is_static()) {
                const auto rank = concat_ptr->get_output_partial_shape(0).rank().get_length();
                axis = ov::util::normalize(concat_ptr->get_axis(), rank);
            }

            auto inputs = concat_ptr->inputs();
            std::map<int64_t, ov::Mask::Ptr> input_masks;
            std::map<int64_t, ov::Mask*> input_masks_row;
            std::vector<int64_t> input_sizes;

            size_t first_input_idx = 0;
            ov::Mask::Ptr first_input_mask;
            bool first_initialized = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                auto input = inputs[i];
                auto input_mask = getMask(input.get_source_output());
                if (input_mask) {
                    input_masks[i] = input_mask;
                    input_masks_row[i] = input_mask.get();

                    if (!first_initialized) {
                        first_input_idx = i;
                        first_input_mask = input_mask;
                        first_initialized = true;
                    }
                }
                input_sizes.push_back(input.get_shape().at(axis));
            }

            if (!first_initialized) {
                return false;
            }

            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();

            auto out_mask_callback = [input_masks_row, input_sizes, axis](ov::Mask::Ptr cur_mask) -> bool {
                int64_t cur_size = 0;
                cur_mask->at(axis).clear();

                for (size_t i = 0; i < input_sizes.size(); ++i) {
                    if (input_masks_row.count(i)) {
                        for (const auto idx : input_masks_row.at(i)->at(axis)) {
                            cur_mask->at(axis).insert(idx + cur_size);
                        }
                    }
                    cur_size += input_sizes[i];
                }
                return true;
            };

            auto create_input_mask_callback_for_idx = [output_mask_row, input_sizes, axis](size_t input_idx) {
                auto input_mask_callback =
                    [output_mask_row, input_sizes, axis, input_idx](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->clean_dim_values();
                    uint64_t min_val = 0;
                    for (size_t i = 0; i < input_idx; i++) {
                        min_val += input_sizes[i];
                    }
                    uint64_t max_val = min_val + input_sizes[input_idx];
                    for (const auto idx : output_mask_row->at(axis)) {
                        if (idx < max_val && idx >= min_val) {
                            cur_mask->at(axis).insert(idx - min_val);
                        }
                    }
                    return true;
                };
                return input_mask_callback;
            };
            output_mask->add_callback(out_mask_callback, first_input_mask);

            for (size_t i = 0; i < inputs.size(); ++i) {
                if (input_masks.count(i) && i != first_input_idx) {
                    auto input_mask = input_masks.at(i);
                    input_mask->add_callback(create_input_mask_callback_for_idx(i), first_input_mask);
                    first_input_mask->add_callback(
                        [](ov::Mask::Ptr cur_mask) -> bool {
                            return true;
                        },
                        input_mask);
                }
            }
            first_input_mask->add_callback(create_input_mask_callback_for_idx(first_input_idx), output_mask);
            output_mask->apply_callback(first_input_mask);
            setMask(m_output, output_mask);

            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(concat, "ConcatMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::PassThrough : public MatcherPass {
public:
    PassThrough() {
        auto unary_op = pattern::wrap_type<op::util::UnaryElementwiseArithmetic,
                                           opset10::Clamp,
                                           opset10::Swish,
                                           opset10::Elu,
                                           opset10::HardSigmoid,
                                           opset10::PRelu,
                                           opset10::Mish,
                                           op::v1::Softmax,
                                           opset10::Softmax,
                                           opset10::SoftPlus,
                                           opset10::Convert,
                                           opset10::ConvertLike,
                                           opset10::AvgPool,
                                           op::v1::MaxPool,
                                           opset10::MaxPool,
                                           opset10::ROIPooling,
                                           opset10::PSROIPooling,
                                           ov::op::util::PadBase,
                                           opset10::MVN,
                                           op::v0::Gelu,
                                           opset10::Gelu>();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_output = pattern_map.at(unary_op);
            const auto& m_input = m_output.get_node_shared_ptr()->input_value(0);

            if (auto input_mask = getMask(m_input)) {
                setMask(m_output, input_mask);
            }

            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(unary_op, "PassThroughMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Reduce : public MatcherPass {
public:
    Reduce() {
        auto inputs = pattern::any_input();
        auto weights = pattern::wrap_type<opset10::Constant>();
        auto pooling_by_reduce =
            pattern::wrap_type<opset10::ReduceMin, opset10::ReduceMax, opset10::ReduceMean>({inputs, weights});

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto m_weights = pattern_map.at(weights);
            const auto& m_input = pattern_map.at(inputs);
            const auto& m_output = pattern_map.at(pooling_by_reduce);

            // Check reduce operation reduces only dimension without masks
            if (auto input_mask = getMask(m_input)) {
                auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
                const auto constant = std::dynamic_pointer_cast<opset10::Constant>(m_weights.get_node_shared_ptr());
                OPENVINO_ASSERT(!!constant, "Dynamic cast returned a nullptr");
                const auto reduce_dims = constant->cast_vector<int64_t>();

                auto input_mask_row = input_mask.get();
                auto output_mask_row = output_mask.get();
                input_mask->add_callback(
                    [output_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                        cur_mask->copy_value_from_mask(output_mask_row);
                        return true;
                    },
                    output_mask);
                output_mask->add_callback(
                    [input_mask_row, reduce_dims](ov::Mask::Ptr cur_mask) -> bool {
                        // Propagate masks through dimension only if this dimension isn't reduced
                        for (size_t dim = 0; dim < std::min(cur_mask->size(), input_mask_row->size()); ++dim)
                            if (std::find(reduce_dims.begin(), reduce_dims.end(), dim) == reduce_dims.end())
                                cur_mask->at(dim) = input_mask_row->at(dim);
                            else if (cur_mask->at(dim) != input_mask_row->at(dim))
                                cur_mask->initialize_dependencies();
                        return true;
                    },
                    input_mask);

                // Invalidate current mask and its parent masks
                output_mask->apply_callback(input_mask);
                setMask(m_output, output_mask);
            }

            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(pooling_by_reduce, "PassThroughReduceMaskPropagation");
        register_matcher(m, callback);
    }
};

using dims_vec = std::vector<size_t>;
static std::vector<dims_vec> map_reshaped_dimensions(const dims_vec input_shape, const dims_vec output_shape) {
    auto dims_map = std::vector<dims_vec>();
    auto cur_output_dims = dims_vec();
    for (size_t i(0), j(0); i < input_shape.size(); ++i) {
        size_t accum(1);
        while (j < output_shape.size()) {
            accum *= output_shape[j];
            cur_output_dims.push_back(j);
            j++;
            if (accum >= input_shape[i])
                break;
        }
        if (accum != input_shape[i])
            break;
        dims_map.push_back(cur_output_dims);
        cur_output_dims.clear();
    }
    return dims_map;
}

static std::vector<ov::Shape> map_reshaped_shapes(const ov::Shape unsquized_shape,
                                                  const std::vector<dims_vec> dims_map) {
    auto retval = std::vector<ov::Shape>();
    for (const auto& unsquized_dims : dims_map) {
        auto cur_dim_shape = ov::Shape();
        for (const auto& dim : unsquized_dims)
            cur_dim_shape.push_back(unsquized_shape[dim]);
        retval.push_back(cur_dim_shape);
    }
    return retval;
}

/* Attributes of unsquized dimension. Channel block is all elements
 *  which have equal coordinates in first k dimensions where k is
 *  a coordinate of the unsquized dimension.
 */
struct DimsAttr {
    size_t elems_inner_dims;  // Amount of elements in each channel block
    size_t elems_outer_dims;  // Amount of channel blocks
    size_t shift;             // Distance between two neigboring channel blocks
    size_t dim;               // Elements in dimension
};

/* Map between squized and unsquized dimensions.
 */
struct ChannelsMap {
    std::set<uint64_t> squized_mask;
    std::map<uint64_t, std::set<uint64_t>> unsquized_mask;
    bool should_init;

    ChannelsMap(std::set<uint64_t>&& p_squized_mask,
                std::map<uint64_t, std::set<uint64_t>>&& p_unsquized_mask,
                bool p_should_init)
        : squized_mask(p_squized_mask),
          unsquized_mask(p_unsquized_mask),
          should_init(p_should_init) {}
};

/* Returns coordinate iterator through all values of given channel
 *  on unsquized_shape_dim dimension according to unsquized_shape shape.
 */
static ov::CoordinateTransformBasic get_channel_iter(const ov::Shape& unsquized_shape,
                                                     const size_t unsquized_shape_dim) {
    auto iter_shape = unsquized_shape;
    iter_shape[unsquized_shape_dim] = 1;
    return ov::CoordinateTransformBasic{iter_shape};
}

/* Maps squzed_mask_dim mask dimension to vector of masks for unsquized_dims.
 *  Using dims_attrs and unsquized_shape for channel iteration.
 */
static ChannelsMap map_channels(const std::set<uint64_t> squized_mask_dim,
                                const dims_vec unsquized_dims,
                                const std::vector<DimsAttr> dims_attrs,
                                const ov::Shape unsquized_shape) {
    auto squized_mask_res = std::set<uint64_t>();
    auto unsquized_mask = std::map<uint64_t, std::set<uint64_t>>();
    auto suspicious_elems = std::set<uint64_t>();
    for (const auto unsquized_dim : unsquized_dims) {
        unsquized_mask[unsquized_dim] = std::set<uint64_t>();
        auto squized_mask_dim_copy = std::set<uint64_t>();
        const auto unsquized_shift = unsquized_dim - unsquized_dims[0];
        std::copy(squized_mask_dim.begin(),
                  squized_mask_dim.end(),
                  std::inserter(squized_mask_dim_copy, squized_mask_dim_copy.begin()));
        while (squized_mask_dim_copy.size()) {
            auto cur_ch_elems = std::set<uint64_t>();
            // Take first element, calculate its
            // channel correspondend to unsquized_dim and try to
            // check all corresponded channels elems
            // are present.
            const auto elem = *squized_mask_dim_copy.begin();
            auto ch = elem / dims_attrs[unsquized_dim].elems_inner_dims;
            if (dims_attrs[unsquized_dim].elems_outer_dims != 1)
                ch %= dims_attrs[unsquized_dim].dim;

            // Start iterating through chanel
            auto iter = get_channel_iter(unsquized_shape, unsquized_shift);
            for (auto coord : iter) {
                coord[unsquized_shift] = ch;
                const auto idx = coordinate_index(coord, unsquized_shape);
                if (squized_mask_dim_copy.find(idx) != squized_mask_dim_copy.end()) {
                    cur_ch_elems.insert(idx);
                    squized_mask_dim_copy.erase(idx);
                }
            }

            if (cur_ch_elems.size() !=
                dims_attrs[unsquized_dim].elems_inner_dims * dims_attrs[unsquized_dim].elems_outer_dims) {
                suspicious_elems.insert(cur_ch_elems.begin(), cur_ch_elems.end());
                continue;
            }
            auto tmp = std::set<uint64_t>();
            std::set_union(squized_mask_res.begin(),
                           squized_mask_res.end(),
                           cur_ch_elems.begin(),
                           cur_ch_elems.end(),
                           std::inserter(tmp, tmp.begin()));
            squized_mask_res = tmp;
            unsquized_mask[unsquized_dim].insert(ch);
        }
    }
    // Check suspicious dims
    auto should_init = false;
    auto diff = std::set<uint64_t>();
    std::set_difference(suspicious_elems.begin(),
                        suspicious_elems.end(),
                        squized_mask_res.begin(),
                        squized_mask_res.end(),
                        std::inserter(diff, diff.begin()));
    if (diff.size())
        should_init = true;
    return ChannelsMap(std::move(squized_mask_res), std::move(unsquized_mask), should_init);
}

/* Collects dimensions attributes according to
 * dims_map map vector and unsquized_shape shape.
 */
static std::vector<DimsAttr> collect_dims_attrs(const std::vector<dims_vec> dims_map,
                                                const std::vector<size_t> unsquized_shape) {
    auto dims_attrs = std::vector<DimsAttr>();
    for (size_t squized_dim = 0; squized_dim < dims_map.size(); ++squized_dim) {
        auto unsquized_dims = dims_map[squized_dim];
        for (size_t in_idx = 0; in_idx < unsquized_dims.size(); ++in_idx) {
            size_t elems_outer_dims = ov::shape_size(unsquized_shape.begin() + unsquized_dims[0],
                                                     unsquized_shape.begin() + unsquized_dims[0] + in_idx);
            size_t elems_inner_dims =
                ov::shape_size(unsquized_shape.begin() + unsquized_dims[0] + in_idx + 1,
                               unsquized_shape.begin() + unsquized_dims[0] + unsquized_dims.size());
            const auto dim = unsquized_shape[unsquized_dims[in_idx]];
            dims_attrs.push_back(DimsAttr{elems_inner_dims, elems_outer_dims, dim * elems_inner_dims, dim});
        }
    }
    return dims_attrs;
}

class ov::pass::mask_propagation::Reshape : public MatcherPass {
public:
    Reshape() {
        auto inputs = pattern::any_input(pattern::has_static_shape());
        auto weights = pattern::any_input();
        auto reshape = pattern::wrap_type<opset10::Reshape>({inputs, weights}, pattern::has_static_shape());

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto m_weights = pattern_map.at(weights);
            const auto& m_input = pattern_map.at(inputs);
            const auto& m_output = pattern_map.at(reshape);

            // Check if this reshape is before group convolution
            // In such case this reshape should be processed by GroupConvolutionReshape pass
            for (const auto& inp : m_output.get_target_inputs())
                if (is_type<opset10::GroupConvolution>(inp.get_node()))
                    return true;

            auto constant = std::dynamic_pointer_cast<opset10::Constant>(m_weights.get_node_shared_ptr());
            if (!constant) {
                constant = ov::util::get_constant_from_source(m_weights.get_node_shared_ptr());
                if (!constant) {
                    OPENVINO_DEBUG("Can't process reshape node ",
                                   m_output.get_node()->get_friendly_name(),
                                   " with no constant node ",
                                   m_weights.get_node()->get_friendly_name(),
                                   " as shape input.");
                    return false;
                }
            }

            // Check reshape operation reshape only dimension without masks
            if (auto input_mask = getMask(m_input)) {
                enum ReshapeType { default_, extend, shrink } configuration(ReshapeType::default_);
                auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
                auto weights_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length(), true);

                const auto input_shape = m_input.get_shape();
                const auto output_shape = m_output.get_node()->output(0).get_shape();

                // Check dimensions equality from the begining and allow
                // to propagate masks only for dimensions which equal from the begining
                size_t not_reshaped_dims;
                {
                    size_t i = 0;
                    for (; i < std::min(input_shape.size(), output_shape.size()); ++i) {
                        if (input_shape[i] != output_shape[i])
                            break;
                    }
                    not_reshaped_dims = i;
                }

                auto input_mask_row = input_mask.get();
                auto weights_mask_row = weights_mask.get();
                auto output_mask_row = output_mask.get();

                // Choose correct configuration
                auto dims_map = map_reshaped_dimensions(input_shape, output_shape);
                if (dims_map.size() == input_shape.size()) {
                    configuration = ReshapeType::extend;
                } else {
                    dims_map = map_reshaped_dimensions(output_shape, input_shape);
                    if (dims_map.size() == output_shape.size())
                        configuration = ReshapeType::shrink;
                }

                switch (configuration) {
                case ReshapeType::default_: {
                    // Case when input and output dimensions size are equal from 0 to k dimentions
                    // and dimension k+1 is not exist in input or output shape or
                    // k+1 dimensionss has different size. Masks are proagating as is through
                    // from 0 to k dimensions.
                    // Example: [a, b, c, 3, 2] -> [a, b, c, 2, 3]
                    input_mask->add_callback(
                        [weights_mask_row, not_reshaped_dims](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t dim = 0; dim < cur_mask->size(); ++dim)
                                if (dim < not_reshaped_dims)
                                    cur_mask->at(dim) = weights_mask_row->at(dim);
                                else
                                    cur_mask->at(dim).clear();
                            return true;
                        },
                        weights_mask);
                    weights_mask->add_callback(
                        [input_mask_row, not_reshaped_dims](ov::Mask::Ptr cur_mask) -> bool {
                            // Propagate masks down through dimension only if this dimension isn't reshaped
                            for (size_t dim = 0; dim < std::min(cur_mask->size(), input_mask_row->size()); ++dim)
                                if (dim < not_reshaped_dims)
                                    cur_mask->at(dim) = input_mask_row->at(dim);
                                else if (!input_mask_row->at(dim).empty())
                                    cur_mask->initialize_dependencies();
                            return true;
                        },
                        input_mask);

                    output_mask->add_callback(
                        [weights_mask_row, not_reshaped_dims](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t dim = 0; dim < cur_mask->size(); ++dim)
                                if (dim < not_reshaped_dims)
                                    cur_mask->at(dim) = weights_mask_row->at(dim);
                                else
                                    cur_mask->at(dim).clear();
                            return true;
                        },
                        weights_mask);

                    weights_mask->add_callback(
                        [output_mask_row, not_reshaped_dims](ov::Mask::Ptr cur_mask) -> bool {
                            // Propagate masks up through dimension only if this dimension isn't reshaped
                            for (size_t dim = 0; dim < std::min(cur_mask->size(), output_mask_row->size()); ++dim)
                                if (dim < not_reshaped_dims)
                                    cur_mask->at(dim) = output_mask_row->at(dim);
                                else if (!output_mask_row->at(dim).empty())
                                    cur_mask->initialize_dependencies();
                            return true;
                        },
                        output_mask);
                }; break;
                case ReshapeType::extend: {
                    // Case when the output shape shape is bigger than the input shape and
                    // each input dimension could be mapped to one or several
                    // successive output dimensions.
                    // Example: [a * b, c, d * e] -> [a, b, c, d, e]
                    // Example mapping: 0 -> [0, 1], 1 -> [2], 2 -> [3, 4]
                    const auto dims_attrs = collect_dims_attrs(dims_map, output_shape);
                    const auto dims_shape = map_reshaped_shapes(output_shape, dims_map);
                    input_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t in_dim = 0; in_dim < dims_map.size(); ++in_dim) {
                                cur_mask->at(in_dim).clear();
                                for (const auto out_dim : dims_map[in_dim]) {
                                    const auto unsquized_shift = out_dim - dims_map[in_dim][0];
                                    for (const auto ch : weights_mask_row->at(out_dim)) {
                                        auto iter = get_channel_iter(dims_shape[in_dim], unsquized_shift);
                                        for (auto coord : iter) {
                                            coord[unsquized_shift] = ch;
                                            cur_mask->at(in_dim).insert(coordinate_index(coord, dims_shape[in_dim]));
                                        }
                                    }
                                }
                            }
                            return true;
                        },
                        weights_mask);

                    weights_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t in_dim = 0; in_dim < dims_map.size(); ++in_dim) {
                                const auto map = map_channels(input_mask_row->at(in_dim),
                                                              dims_map[in_dim],
                                                              dims_attrs,
                                                              dims_shape[in_dim]);
                                for (const auto& dim : map.unsquized_mask)
                                    cur_mask->at(dim.first) = dim.second;
                                if (map.should_init)
                                    cur_mask->initialize_dependencies();
                            }
                            return true;
                        },
                        input_mask);

                    output_mask->add_callback(
                        [weights_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                            cur_mask->copy_value_from_mask(weights_mask_row);
                            return true;
                        },
                        weights_mask);

                    weights_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            cur_mask->copy_value_from_mask(output_mask_row);
                            return true;
                        },
                        output_mask);
                }; break;
                case ReshapeType::shrink: {
                    // Case when the input shape shape is bigger than the output shape and
                    // each output dimension could be mapped to one or several
                    // successive input dimensions.
                    // Example: [a, b, c, d, e] -> [a * b, c, d * e]
                    // Example mapping: 0 -> [0, 1], 1 -> [2], 2 -> [3, 4]
                    const auto dims_attrs = collect_dims_attrs(dims_map, input_shape);
                    const auto dims_shape = map_reshaped_shapes(input_shape, dims_map);
                    input_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t out_dim = 0; out_dim < dims_map.size(); ++out_dim) {
                                const auto map = map_channels(weights_mask_row->at(out_dim),
                                                              dims_map[out_dim],
                                                              dims_attrs,
                                                              dims_shape[out_dim]);
                                for (const auto& dim : map.unsquized_mask)
                                    cur_mask->at(dim.first) = dim.second;
                                if (map.should_init)
                                    cur_mask->initialize_dependencies();
                            }
                            return true;
                        },
                        weights_mask);

                    weights_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t out_dim = 0; out_dim < dims_map.size(); ++out_dim) {
                                cur_mask->at(out_dim).clear();
                                for (const auto in_dim : dims_map[out_dim]) {
                                    const auto unsquized_shift = in_dim - dims_map[out_dim][0];
                                    for (const auto ch : input_mask_row->at(in_dim)) {
                                        auto iter = get_channel_iter(dims_shape[out_dim], unsquized_shift);
                                        for (auto coord : iter) {
                                            coord[unsquized_shift] = ch;
                                            cur_mask->at(out_dim).insert(coordinate_index(coord, dims_shape[out_dim]));
                                        }
                                    }
                                }
                            }
                            return true;
                        },
                        input_mask);

                    output_mask->add_callback(
                        [weights_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                            cur_mask->copy_value_from_mask(weights_mask_row);
                            return true;
                        },
                        weights_mask);

                    weights_mask->add_callback(
                        [=](ov::Mask::Ptr cur_mask) -> bool {
                            for (size_t out_dim = 0; out_dim < dims_map.size(); ++out_dim) {
                                const auto map = map_channels(output_mask_row->at(out_dim),
                                                              dims_map[out_dim],
                                                              dims_attrs,
                                                              dims_shape[out_dim]);
                                cur_mask->at(out_dim) = map.squized_mask;
                                if (map.should_init)
                                    cur_mask->initialize_dependencies();
                            }
                            return true;
                        },
                        output_mask);
                }; break;
                }

                weights_mask->apply_callback(input_mask);
                setMask(m_output, output_mask);
                setMask(m_weights, weights_mask);
            }

            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape, "ReshapeMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Transpose : public MatcherPass {
public:
    Transpose() {
        auto input = pattern::any_input();
        auto weights = pattern::any_input();
        auto transpose = pattern::wrap_type<opset10::Transpose>({input, weights});
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_input = pattern_map.at(input);
            const auto& m_weights = pattern_map.at(weights);
            const auto& m_output = pattern_map.at(transpose);

            const auto input_order_node = ov::util::get_constant_from_source(m_weights.get_node_shared_ptr());
            if (!input_order_node) {
                OPENVINO_DEBUG("Can't process transpose node ",
                               m_output.get_node()->get_friendly_name(),
                               " with no constant node ",
                               m_weights.get_node()->get_friendly_name(),
                               " as input_order input.");
                return false;
            }

            const auto input_mask = getMask(m_input);
            if (!input_mask) {
                OPENVINO_DEBUG("No input mask for: ", m_output.get_node()->get_friendly_name(), "\n");
                return false;
            }
            if (static_cast<int64_t>(input_mask->size()) != m_output.get_partial_shape().rank().get_length()) {
                OPENVINO_DEBUG("Transpose which change tensor rank is not supported yet.");
                return false;
            }

            const auto forward_order = input_order_node->cast_vector<int64_t>();
            auto backward_order = std::vector<int64_t>();
            for (size_t i = 0; i < input_mask->size(); ++i) {
                const auto dim = std::find(forward_order.begin(), forward_order.end(), i) - forward_order.begin();
                // Dim should be valid because of transpose operation input_order input restrictions
                backward_order.push_back(dim);
            }
            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
            const auto input_mask_row = input_mask.get();
            const auto output_mask_row = output_mask.get();

            output_mask->add_callback(
                [input_mask_row, forward_order](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->clear();
                    for (const auto dim : forward_order)
                        cur_mask->push_back(input_mask_row->at(dim));
                    return true;
                },
                input_mask);
            input_mask->add_callback(
                [output_mask_row, backward_order](ov::Mask::Ptr cur_mask) -> bool {
                    cur_mask->clear();
                    for (const auto dim : backward_order)
                        cur_mask->push_back(output_mask_row->at(dim));
                    return true;
                },
                output_mask);
            if (!output_mask->apply_callback(input_mask)) {
                return false;
            }
            setMask(m_output, output_mask);
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose, "TransposePropagation");
        register_matcher(m, callback);
    }
};

static ov::Mask::Ptr create_connect_split_output_mask(ov::Mask::Ptr input_mask,
                                                      const int64_t axis,
                                                      const uint64_t split_start,
                                                      const uint64_t split_end) {
    auto output_mask = std::make_shared<ov::Mask>();
    auto input_mask_raw = input_mask.get();
    output_mask->add_callback(
        [input_mask_raw, axis, split_start, split_end](ov::Mask::Ptr cur_mask) -> bool {
            cur_mask->copy_and_slice_mask_from(input_mask_raw, axis, split_start, split_end);
            return true;
        },
        input_mask);
    auto output_mask_raw = output_mask.get();
    input_mask->add_callback(
        [output_mask_raw, axis, split_start, split_end](ov::Mask::Ptr cur_mask) -> bool {
            auto& dim_mask = cur_mask->at(axis);
            auto it = dim_mask.lower_bound(split_start);
            while (it != dim_mask.end() && *it < split_end) {
                it = dim_mask.erase(it);
            }
            for (size_t j = 0; j < output_mask_raw->size(); j++) {
                const auto& dim_mask = output_mask_raw->at(j);
                if (static_cast<int64_t>(j) == axis) {
                    for (const auto d : dim_mask)
                        cur_mask->at(j).insert(d + split_start);
                } else {
                    cur_mask->at(j) = dim_mask;
                }
            }
            return true;
        },
        output_mask);

    return output_mask;
}

class ov::pass::mask_propagation::VariadicSplit : public MatcherPass {
public:
    VariadicSplit() {
        auto input_pattern = pattern::any_input(pattern::has_static_rank());
        auto axis_pattern = pattern::wrap_type<ov::opset10::Constant>();
        auto split_lengths_pattern = pattern::wrap_type<ov::opset10::Constant>();
        auto split_pattern =
            pattern::wrap_type<ov::opset10::VariadicSplit>({input_pattern, axis_pattern, split_lengths_pattern});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto axis_node = as_type<ov::opset10::Constant>(pattern_map.at(axis_pattern).get_node());
            const auto& input = pattern_map.at(input_pattern);
            const auto input_mask = ov::getMask(input);
            auto split = pattern_map.at(split_pattern).get_node();
            auto split_lengths_const = as_type<ov::opset10::Constant>(pattern_map.at(split_lengths_pattern).get_node());

            if (!axis_node)
                return false;
            if (!input_mask)
                return false;
            if (!split_lengths_const)
                return false;

            auto split_lengths = split_lengths_const->cast_vector<int64_t>();
            auto axis = axis_node->cast_vector<int64_t>()[0];
            if (axis < 0)
                axis += input_mask->size();

            // adjust split_lengths if needed
            // split_lengths can contain -1 value
            int minus_one_length_idx = -1;
            int64_t total_lengths = 0;
            for (size_t i = 0; i < split_lengths.size(); i++) {
                if (split_lengths[i] == -1) {
                    minus_one_length_idx = static_cast<int>(i);
                    continue;
                }
                total_lengths += split_lengths[i];
            }
            if (minus_one_length_idx >= 0 && !input_mask->at(axis).empty()) {
                const auto& input_shape = input.get_partial_shape();
                if (input_shape[axis].is_dynamic())
                    return false;
                auto split_dim = input_shape[axis].get_length();
                split_lengths[minus_one_length_idx] = split_dim - total_lengths;
            }

            uint64_t split_start = 0;
            uint64_t split_end = 0;
            std::vector<ov::Mask::Ptr> output_masks;
            for (size_t i = 0; i < split->get_output_size(); i++) {
                split_end += split_lengths[i];
                output_masks.push_back(create_connect_split_output_mask(input_mask, axis, split_start, split_end));
                ov::setMask(split->output(i), output_masks[i]);
                split_start = split_end;
            }
            for (const auto& output_mask : output_masks) {
                output_mask->apply_callback(input_mask);
            }
            return true;
        };
        auto m = std::make_shared<pattern::Matcher>(split_pattern, "VariadicSplitMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::Split : public MatcherPass {
public:
    Split() {
        auto input_pattern = pattern::any_input(pattern::has_static_rank());
        auto axis_pattern = pattern::wrap_type<ov::opset10::Constant>();
        auto split_pattern = pattern::wrap_type<ov::opset10::Split>({input_pattern, axis_pattern});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto axis_node = as_type<ov::opset10::Constant>(pattern_map.at(axis_pattern).get_node());
            const auto& input = pattern_map.at(input_pattern);
            const auto input_mask = ov::getMask(input);

            if (!axis_node)
                return false;

            if (!input_mask)
                return false;

            auto axis = axis_node->cast_vector<int64_t>()[0];
            if (axis < 0)
                axis += input_mask->size();

            const auto& input_shape = input.get_partial_shape();
            if (input_shape[axis].is_dynamic())
                return false;
            const auto& split = pattern_map.at(split_pattern).get_node();
            auto num_splits = split->get_output_size();
            auto split_dim = static_cast<uint64_t>(input_shape[axis].get_length());

            uint64_t split_start = 0;
            auto split_step = split_dim / num_splits;
            uint64_t split_end = split_step;
            std::vector<ov::Mask::Ptr> output_masks;
            for (size_t i = 0; i < split->get_output_size(); i++) {
                output_masks.push_back(create_connect_split_output_mask(input_mask, axis, split_start, split_end));
                ov::setMask(split->output(i), output_masks[i]);
                split_start = split_end;
                split_end += split_step;
            }
            for (const auto& output_mask : output_masks) {
                output_mask->apply_callback(input_mask);
            }
            return true;
        };
        auto m = std::make_shared<pattern::Matcher>(split_pattern, "SplitMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::StopPropagation : public MatcherPass {
public:
    StopPropagation() {
        auto any_node = pattern::any_input();

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            const auto& m_output = pattern_map.at(any_node);
            const auto& node = m.get_match_root();

            auto output_mask = std::make_shared<ov::Mask>(m_output.get_partial_shape().rank().get_length());
            auto output_mask_row = output_mask.get();
            bool any_input_with_masks = false;
            for (const auto& input : node->input_values()) {
                if (auto input_mask = getMask(input)) {
                    auto input_mask_row = input_mask.get();
                    input_mask->add_callback(
                        [output_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                            cur_mask->clean_dim_values();
                            if (!output_mask_row->all_dims_are_empty())
                                cur_mask->initialize_dependencies();
                            return true;
                        },
                        output_mask);
                    output_mask->add_callback(
                        [input_mask_row](ov::Mask::Ptr cur_mask) -> bool {
                            cur_mask->copy_value_from_mask(input_mask_row);
                            return true;
                        },
                        input_mask);

                    // Invalidate current mask and its parent masks
                    output_mask->apply_callback(input_mask);
                    OPENVINO_DEBUG("Invalidate masks for ",
                                   *input.get_node(),
                                   " because ",
                                   node,
                                   " is in scope of stop ops.\n");
                    any_input_with_masks = true;
                }
            }
            if (any_input_with_masks) {
                // Set mask to stop op first input tensor to prevent mask rewriting for
                // nodes which share output tensor with previous node.
                if (ov::is_type<opset10::Result>(m_output.get_node_shared_ptr()))
                    setMask(*m_output.get_node()->inputs().begin(), output_mask);
                else
                    setMask(m_output, output_mask);
            }
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(any_node, "StopMaskPropagation");
        register_matcher(m, callback);
    }
};

class ov::pass::mask_propagation::SkipPropagation : public MatcherPass {
public:
    SkipPropagation() {
        // Skip mask propagation for ShapeOf operation to prevent this opearation to be
        // processed as stop op.
        auto node = pattern::wrap_type<opset10::ShapeOf, op::v0::ShapeOf>();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(node, "SkipPropagation");
        register_matcher(m, callback);
    }
};

ov::pass::PropagateMasks::PropagateMasks() {
    add_matcher<mask_propagation::MatMul>();
    add_matcher<mask_propagation::Convolution>();
    add_matcher<mask_propagation::GroupConvolutionReshape>();
    add_matcher<mask_propagation::GroupConvolution>();
    add_matcher<mask_propagation::Elementwise>();
    add_matcher<mask_propagation::PassThrough>();
    add_matcher<mask_propagation::Reduce>();
    add_matcher<mask_propagation::Reshape>();
    add_matcher<mask_propagation::Transpose>();
    add_matcher<mask_propagation::FakeQuantize>();
    add_matcher<mask_propagation::Concat>();
    add_matcher<ov::pass::mask_propagation::VariadicSplit>();
    add_matcher<ov::pass::mask_propagation::Split>();
    add_matcher<mask_propagation::SkipPropagation>();
    add_matcher<mask_propagation::StopPropagation>();
}
