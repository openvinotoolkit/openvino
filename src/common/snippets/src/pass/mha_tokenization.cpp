// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mha_tokenization.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/util/pp.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/utils/tokenization_utils.hpp"
#include "snippets/utils/utils.hpp"

using namespace ov::snippets::utils;

namespace {
bool is_supported_tensor(const ov::descriptor::Tensor& t) {
    return t.get_partial_shape().rank().is_static() && any_of(t.get_partial_shape().size(), 2LU, 3LU, 4LU);
}

bool is_supported_intermediate_op(const std::shared_ptr<ov::Node>& node) {
    const auto is_intermediate_op = [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                                  ov::op::util::BinaryElementwiseArithmetic,
                                  ov::op::v0::FakeQuantize,
                                  ov::op::v1::Select>(node);
    };
    return is_intermediate_op(node) && ov::snippets::pass::TokenizeSnippets::AppropriateForSubgraph(node);
}

bool is_valid_transpose(const std::shared_ptr<ov::opset1::Transpose>& node,
                        const std::set<size_t>& supported_ranks,
                        const std::vector<int32_t>& expected_order) {
    auto is_valid_transpose_order = [expected_order, supported_ranks](const std::shared_ptr<ov::Node>& node) -> bool {
        const auto transpose_pattern = ov::as_type_ptr<ov::opset1::Constant>(node);
        if (!transpose_pattern) {
            return false;
        }
        const auto existing_order = transpose_pattern->cast_vector<int32_t>();
        return existing_order == expected_order && supported_ranks.count(existing_order.size()) != 0;
    };
    auto is_supported_transpose_tensor = [](const ov::descriptor::Tensor& t) {
        return is_supported_tensor(t) &&
               ov::snippets::pass::TokenizeSnippets::get_supported_element_types().count(t.get_element_type()) != 0;
    };

    return node && node->get_output_target_inputs(0).size() == 1 &&
           is_valid_transpose_order(node->get_input_node_shared_ptr(1)) &&
           is_supported_transpose_tensor(node->get_input_tensor(0));
}

void tokenize_broadcast(const std::shared_ptr<ov::Node>& interm_op, ov::NodeVector& ordered_ops) {
    // We can tokenize Broadcast op only when output shape of child doesn't depend on Broadcast shape without last
    // dimension. Snippets remove Broadcast op and insert BroadcastMove if last dimensions before and after Broadcast
    // are different. Otherwise, we can lose original shape. Example:
    //        in0 [1, 1, 1]      in0 [1, 1, 1]              in0 [1, 1, 1]   in0 [1, 1, 1]
    //     Broadcast [1, 10, 1]    /                                 \       /
    //           \               /                --->>>                Add
    //                  Add                                              |
    //             Result [1, 10, 1]                              Result [1, 1, 1]

    ov::PartialShape new_output_shape(std::vector<ov::Dimension>{1});
    ov::NodeVector broadcast_nodes;

    auto skip_last_dim = [](const ov::PartialShape& shape) {
        return ov::PartialShape(std::vector<ov::Dimension>{shape.begin(), shape.end() - 1});
    };

    for (auto input : interm_op->inputs()) {
        auto broadcast = ov::as_type_ptr<ov::opset1::Broadcast>(input.get_source_output().get_node_shared_ptr());
        // TODO: Can we reuse AppropriateForSubgraph here? Seems like it's huge check for Broadcast
        if (broadcast && broadcast->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY &&
            broadcast->get_output_target_inputs(0).size() == 1) {
            // TODO: Add support of Broadcast with ShapeOf subgraph on second input
            if (!ov::is_type<ov::op::v0::Constant>(broadcast->input_value(1).get_node_shared_ptr())) {
                continue;
            }

            broadcast_nodes.push_back(broadcast);

            const auto pshape = broadcast->get_input_partial_shape(0);
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ov::op::AutoBroadcastType::NUMPY);
            }
        } else {
            const auto& pshape = input.get_partial_shape();
            if (pshape.rank().is_static() && pshape.size() > 2) {
                ov::PartialShape::broadcast_merge_into(new_output_shape,
                                                       skip_last_dim(pshape),
                                                       ::ov::op::AutoBroadcastType::NUMPY);
            }
        }
    }

    if (!broadcast_nodes.empty()) {
        if (new_output_shape == skip_last_dim(interm_op->get_output_partial_shape(0))) {
            std::copy(broadcast_nodes.begin(), broadcast_nodes.end(), std::back_inserter(ordered_ops));
        }
    }
}

bool tokenize_reshape_around_softmax(std::shared_ptr<ov::Node>& interm_op,
                                     std::shared_ptr<ov::opset1::Reshape>& reshape,
                                     ov::NodeVector& ordered_ops) {
    reshape = ov::as_type_ptr<ov::opset1::Reshape>(interm_op);
    if (reshape) {
        // TODO: Add support of Reshape with ShapeOf subgraph on second input
        if (!ov::is_type<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr())) {
            return false;
        }

        const auto in_shape = reshape->get_input_partial_shape(0);
        const auto out_shape = reshape->get_output_partial_shape(0);
        const auto in_last_dim = *in_shape.crbegin();
        const auto out_last_dim = *out_shape.crbegin();
        if (in_last_dim.is_dynamic() || out_last_dim.is_dynamic() || in_last_dim != out_last_dim ||
            reshape->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        ordered_ops.push_back(reshape);
        interm_op = reshape->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
}

bool update_intermediate_supported_ops(std::shared_ptr<ov::Node>& interm_op,
                                       ov::NodeVector& ordered_ops,
                                       size_t& n_potential_body_params) {
    while (is_supported_intermediate_op(interm_op)) {
        // All supported intermediate ops have only one output port
        if (interm_op->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        // Check for supported ops on branches: Broadcast/Elementwise (for example, dequantize ops)
        if (interm_op->get_input_size() > 1) {
            tokenize_broadcast(interm_op, ordered_ops);
            auto is_supported_branch_op = [&ordered_ops](const std::shared_ptr<ov::Node>& op) {
                return is_supported_intermediate_op(op) &&
                       ov::snippets::pass::GetSnippetsNodeType(op) !=
                           ov::snippets::pass::SnippetsNodeType::SkippedByPlugin &&
                       std::find(ordered_ops.begin(), ordered_ops.end(), op) == ordered_ops.end();
            };

            for (size_t i = 0; i < interm_op->get_input_size(); ++i) {
                const size_t shift = ordered_ops.size();
                auto parent = interm_op->get_input_node_shared_ptr(i);
                while (is_supported_branch_op(parent)) {
                    // All supported ops have only one output port
                    if (parent->get_output_target_inputs(0).size() != 1) {
                        break;
                    }

                    // Add node only if there are scalar constants on inputs because of plugin-specific limitation
                    if (!ov::snippets::pass::ExplicitTransposeMatMulInputs::are_weights_scalar(parent)) {
                        break;
                    }

                    ordered_ops.insert(ordered_ops.begin() + shift, parent);
                    // TODO [107731]: We think that sequence of ops goes through input port 0
                    //                But can be Select here? If it can be, parent shouldn't be on input port 0. Need
                    //                another way?
                    if (parent->get_input_size() > 0) {
                        parent = parent->get_input_node_shared_ptr(0);
                    }
                }
            }
        }

        n_potential_body_params += get_potential_body_params(interm_op);

        ordered_ops.push_back(interm_op);
        interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    }
    return true;
}

std::vector<int32_t> get_rank_equivalent_order(std::vector<int32_t> default_order, size_t rank) {
    assert(rank > 2 && "Incorrect order rank for Transpose tokenization");
    auto order = std::vector<int32_t>(rank);
    std::iota(order.begin(), order.end(), 0);
    const auto diff = static_cast<int32_t>(rank - default_order.size());
    for (size_t i = 0; i < default_order.size(); ++i) {
        order[diff + i] = default_order[i] + diff;
    }
    return order;
}
}  // namespace

std::vector<int32_t> ov::snippets::pass::TokenizeMHASnippets::get_fusion_transpose_order(size_t rank) {
    return get_rank_equivalent_order({1, 0, 2}, rank);
}
std::vector<int32_t> ov::snippets::pass::TokenizeMHASnippets::get_decomposed_transpose_order(size_t rank) {
    return get_rank_equivalent_order({1, 2, 0}, rank);
}

bool ov::snippets::pass::TokenizeMHASnippets::is_matmul0_supported(const std::shared_ptr<ov::opset1::MatMul>& matmul) {
    if (!matmul || matmul->get_output_target_inputs(0).size() != 1 || matmul->get_transpose_a() ||
        !is_supported_tensor(matmul->get_input_tensor(0)) || !is_supported_tensor(matmul->get_input_tensor(1))) {
        return false;
    }

    const auto matmul_prc =
        op::Brgemm::get_output_type(matmul->get_input_element_type(0), matmul->get_input_element_type(1));
    return matmul_prc != element::dynamic;
}

ov::snippets::pass::TokenizeMHASnippets::TokenizeMHASnippets(const Config& config) {
    MATCHER_SCOPE(TokenizeMHASnippets);

    auto m_matmul0 =
        std::make_shared<ov::opset1::MatMul>(ov::pass::pattern::any_input(), ov::pass::pattern::any_input());

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(m_matmul0, matcher_name),
        [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMHASnippets")
            auto& pattern_to_output = m.get_pattern_value_map();

            // Queries + Key + Values = 3 standard inputs of MHA
            // Also, some Constants may be extracted outside of Subgraph body,
            // so they should be taken into account as parameters
            size_t n_potential_body_params = 3;
            // The count of potential unique Buffers - it's hidden virtual ports as well
            // We should go through Subgraph and calculate potential non-inplace Buffers count.
            // Example:
            //     Buffer - i32 [32, 128] -> ~ Loop ~ -> Buffer - i8 [32, 128]
            //     After each Loop iteration we should increment pointers of Buffers: accordingly on 4 byte and 1 byte
            //     for scalar case. It means that these increments are not proportional => Each Buffer should have the
            //     own register
            // For that we can just check the following "branches":
            //  - Between MatMul0 and MatMul1:
            //        Softmax is sync point. The operations between MatMul0 -> Softmax and Softmax -> MatMul1
            //        will be fused into one loop after conversion to snippet dialect (Because it's just FQ, Eltwise
            //        nodes)
            //  - Between MatMul0 and Transpose1:
            //        At the moment operations after Transpose1 cannot be fused in inner Transpose Loop
            //        (to avoid performance regressions due to scalar calculations).
            //        But operations after Transpose1 and before MatMul0  will be fused into one loop as well (look at
            //        first point)
            size_t n_buffer_reg_groups = 1;  // After MatMul0 there is always one Buffer
            // Loop depth could reach 3 because of SplitLoops optimization
            static constexpr size_t n_loops_depth = 3;
            ov::NodeVector ordered_ops;

            /* Skeleton on MHA-pattern is:
             *              \     /
             *              MatMul0
             *                 |
             *    Eltwise/Select/Reshape/FakeQuantize
             *                 |
             *              Softmax
             *                 |
             *    Eltwise/Select/Reshape/FakeQuantize
             *                  \      /
             *                   MatMul1
             */
            const auto matmul0 =
                ov::as_type_ptr<ov::opset1::MatMul>(pattern_to_output.at(m_matmul0).get_node_shared_ptr());
            if (!is_matmul0_supported(matmul0)) {
                return false;
            }

            ordered_ops.push_back(matmul0);

            const auto pattern_rank = matmul0->get_output_partial_shape(0).size();

            auto interm_op = matmul0->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            // Add supported operations which are between MatMul0 and Softmax to ordered_ops
            if (!update_intermediate_supported_ops(interm_op, ordered_ops, n_potential_body_params)) {
                return false;
            }

            std::shared_ptr<ov::opset1::Reshape> reshape0 = nullptr;
            if (!tokenize_reshape_around_softmax(interm_op, reshape0, ordered_ops)) {
                return false;
            }

            int64_t axis = 0;
            const auto rank = interm_op->get_input_partial_shape(0).rank();
            if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(interm_op)) {
                axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *interm_op);
            } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(interm_op)) {
                axis = softmax_v1->get_axis();
            } else {
                return false;
            }

            if (axis != rank.get_length() - 1 || interm_op->get_output_target_inputs(0).size() != 1) {
                return false;
            }

            ordered_ops.push_back(interm_op);

            interm_op = interm_op->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            std::shared_ptr<ov::opset1::Reshape> reshape1 = nullptr;
            if (!tokenize_reshape_around_softmax(interm_op, reshape1, ordered_ops)) {
                return false;
            }

            if (((reshape0 == nullptr) != (reshape1 == nullptr)) ||
                (reshape0 && reshape1 &&
                 (reshape0->get_input_partial_shape(0) != reshape1->get_output_partial_shape(0)))) {
                return false;
            }

            // Add supported operations which are between Softmax and MatMul1 to ordered_ops
            if (!update_intermediate_supported_ops(interm_op, ordered_ops, n_potential_body_params)) {
                return false;
            }

            const auto matmul1 = ov::as_type_ptr<ov::opset1::MatMul>(interm_op);
            if (!matmul1 || matmul1->get_transpose_a() || matmul1->get_transpose_b()) {
                return false;
            }

            const auto matmul1_out_type =
                op::Brgemm::get_output_type(matmul1->get_input_element_type(0), matmul1->get_input_element_type(1));
            if (matmul1_out_type == element::dynamic || !is_supported_tensor(matmul1->get_input_tensor(0)) ||
                !is_supported_tensor(matmul1->get_input_tensor(1))) {
                return false;
            }

            if (transformation_callback(matmul0)) {
                return false;
            }

            // Between Softmax and MatMul1 will be the one Loop because of LoopFusing optimization.
            // The Loop will have one Buffer with the same shape both on input and output.
            // Need to check for precision to get if we need one more register for Buffer
            const auto matmul0_prc =
                op::Brgemm::get_output_type(matmul0->get_input_element_type(0), matmul0->get_input_element_type(1));
            if (matmul1->get_input_element_type(0).size() != matmul0_prc.size()) {
                n_buffer_reg_groups++;
            }

            /***** Transposes *****/
            /* There may be Transpose and Reshape ops on inputs and outputs of MHA-pattern skeleton
             * We can add them into Subgraph body
             *       Transpose0  Transpose1
             *              \     /
             *              MatMul0
             *                 |
             *               [...]   Transpose2
             *                  \      /
             *                   MatMul1
             *                      |
             *                  Transpose3
             */

            auto tokenize_transpose = [&](const std::shared_ptr<ov::opset1::Transpose>& transpose,
                                          bool is_input_transposed,
                                          size_t rank,
                                          const ov::NodeVector::const_iterator& pos) {
                // Transpose is redundant for rank <= 2
                if (rank <= 2) {
                    return;
                }
                // If Transpose has valid order for the Transpose fusing (ExplicitTransposeMatMulInputs pass call),
                // tokenize him. Otherwise, skip the Transpose.
                auto order = get_fusion_transpose_order(rank);
                if (!is_input_transposed) {
                    if (is_valid_transpose(transpose, config.get_mha_supported_transpose_ranks(), order)) {
                        ordered_ops.insert(pos, transpose);
                    }
                    return;
                }
                std::swap(order[rank - 1], order[rank - 2]);
                if (is_valid_transpose(transpose, config.get_mha_supported_transpose_ranks(), order)) {
                    ordered_ops.insert(pos, transpose);
                }
            };

            // [160177]: Due to performance problems, if operations on 2nd input of MatMuls should be explicitly
            // executed
            //          (in other words, if the Buffer should be inserted between Brgemm and this op sequence),
            //          we don't tokenize such operations into Subgraph. The details are described in the ticket 160177.
            //          Please, return the tokenization of these ops when parallel loops are implemented.
            const auto transpose0 = ov::as_type_ptr<ov::opset1::Transpose>(matmul0->get_input_node_shared_ptr(0));
            const auto transpose1 = ov::as_type_ptr<ov::opset1::Transpose>(matmul0->get_input_node_shared_ptr(1));
            const auto transpose2 = ov::as_type_ptr<ov::opset1::Transpose>(matmul1->get_input_node_shared_ptr(1));
            tokenize_transpose(transpose0, matmul0->get_transpose_a(), pattern_rank, ordered_ops.begin());
            tokenize_transpose(transpose1, matmul0->get_transpose_b(), pattern_rank, ordered_ops.begin());
            tokenize_transpose(transpose2, matmul1->get_transpose_b(), pattern_rank, ordered_ops.end());
            ordered_ops.push_back(matmul1);

            bool are_ops_after_matmul1 = false;
            auto child = matmul1->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            const auto can_be_ops_after_matmul1_tokenized = matmul1->get_output_target_inputs(0).size() == 1;
            bool has_matmul1_has_ops_on_output = false;
            while (can_be_ops_after_matmul1_tokenized && is_supported_intermediate_op(child)) {
                are_ops_after_matmul1 = true;
                // All supported ops have only one output port
                if (child->get_output_target_inputs(0).size() != 1) {
                    break;
                }

                n_potential_body_params += get_potential_body_params(child);
                const size_t n_child_consumers = child->get_output_target_inputs(0).size();
                // TODO [75567]: move this plugin-specific constraint to the plugin callback
                if (!config.is_gprs_count_sufficient(n_potential_body_params + n_child_consumers,
                                                     n_buffer_reg_groups,
                                                     n_loops_depth)) {
                    break;
                }

                ordered_ops.push_back(child);
                child = child->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
                has_matmul1_has_ops_on_output = true;
            }
            if (has_matmul1_has_ops_on_output) {
                n_buffer_reg_groups++;
            }

            // At the moment Snippets don't support nodes between MatMul1 and Transpose3 due to Loop and strided
            // calculations limitations
            //     MatMul1
            //  <Supported ops>
            //    Transpose3
            if (can_be_ops_after_matmul1_tokenized && !are_ops_after_matmul1 && pattern_rank > 2) {
                auto transpose3 = config.get_mha_token_enable_transpose_on_output()
                                      ? ov::as_type_ptr<ov::opset1::Transpose>(child)
                                      : nullptr;
                if (is_valid_transpose(transpose3,
                                       config.get_mha_supported_transpose_ranks(),
                                       get_fusion_transpose_order(pattern_rank)) &&
                    transpose3->get_input_element_type(0) ==
                        matmul1_out_type) {  // To avoid Convert between MatMul1 and Transpose3
                    ordered_ops.push_back(transpose3);
                }
            }

            /* ======= Plugin related checks ========= */
            const bool is_dynamic =
                std::any_of(ordered_ops.cbegin(), ordered_ops.cend(), [](const std::shared_ptr<Node>& n) {
                    return n->is_dynamic();
                });
            if (is_dynamic && !config.is_dynamic_mha_token_enabled()) {
                return false;
            }

            // TODO [75567]: move this plugin-specific constraint to the plugin callback
            const auto io_count = n_potential_body_params + ordered_ops.back()->get_output_size();
            if (!config.is_gprs_count_sufficient(io_count, n_buffer_reg_groups, n_loops_depth, is_dynamic)) {
                return false;
            }

            const auto subgraph = tokenize_ordered_nodes(ordered_ops);
            // mark the Subgraph as Completed to not allow Snippets to include any nodes into the MHA Subgraph in common
            // Tokenization
            SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);

            return true;
        });
}
