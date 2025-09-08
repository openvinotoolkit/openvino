// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_binary.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

TSBinaryForward::TSBinaryForward() : TSForwardBase() {
    MATCHER_SCOPE(TSBinaryForward);
    create_pattern<op::util::BinaryElementwiseArithmetic,
                   op::util::BinaryElementwiseComparison,
                   op::util::BinaryElementwiseLogical,
                   ov::op::v0::PRelu,
                   ov::op::v0::FakeQuantize>();
    transpose_sinking(matcher_name);
}

namespace {
using NodePtr = std::shared_ptr<Node>;
/**
 * Inserts Unsqueeze node as a child to @arg node with axes {0, 2, ... N - 1}, where N = @arg n_dims
 */
NodePtr InsertBroadcastUnsqueezePReluSlope(const Output<Node>& node, size_t n_dims) {
    if (!n_dims)
        return node.get_node_shared_ptr();

    std::vector<size_t> dims(n_dims);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    auto unsqueeze_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, Shape{dims.size()}, dims);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(node, unsqueeze_const);
    copy_runtime_info(node.get_node_shared_ptr(), {unsqueeze, unsqueeze_const});
    return unsqueeze;
}

/**
 * PRelu has special case. If slope input rank is 1 and its dimension is equal to
 * the second dimension of data input, then per channel broadcast is applied.
 * In such a case we need to insert Unsqueeze before Transpose with another axes.
 */
bool IsSpecialPRelu(NodePtr node) {
    auto prelu = as_type_ptr<ov::op::v0::PRelu>(node);
    if (!prelu)
        return false;

    const auto& slope_shape = prelu->get_input_partial_shape(1);
    if (slope_shape.size() != 1)
        return false;
    const auto& slope_channel_dim = slope_shape[0];
    if (slope_channel_dim.is_dynamic())
        return false;

    const auto& arg_shape = prelu->get_input_partial_shape(0);
    if (arg_shape.rank().is_dynamic())
        return false;

    const auto& channel_dim_idx = arg_shape.size() > 1 ? 1 : 0;
    const auto& arg_channel_dim = arg_shape[channel_dim_idx];
    if (arg_channel_dim.is_dynamic())
        return false;

    return arg_channel_dim == slope_channel_dim;
}
}  // namespace

TSBinaryBackward::TSBinaryBackward() {
    MATCHER_SCOPE(TSBinaryBackward);

    auto main_node_label = wrap_type<op::util::BinaryElementwiseArithmetic,
                                     op::util::BinaryElementwiseComparison,
                                     op::util::BinaryElementwiseLogical,
                                     ov::op::v0::PRelu,
                                     ov::op::v0::FakeQuantize>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
            as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        auto InsertUnsqueeze =
            IsSpecialPRelu(main_node) ? InsertBroadcastUnsqueezePReluSlope : InsertBroadcastUnsqueeze;
        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                                                                       /* input_indexes */ {},
                                                                       InsertUnsqueeze)) {
            register_new_node(new_node);
        }
        main_node->validate_and_infer_types();
        RemoveTransposeConsumers(main_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
