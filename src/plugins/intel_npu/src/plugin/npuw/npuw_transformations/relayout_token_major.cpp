// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "relayout_token_major.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

bool is_token_major(const ov::PartialShape& shape, std::size_t tokens) {
    return shape.is_static() && shape.rank().get_length() >= 3 &&
           static_cast<std::size_t>(shape[0].get_length()) == tokens;
}

// An operand whose dimensions are all 1 but the innermost broadcasts against any
// token count, so the rewrite leaves it as is.
bool broadcasts_over_tokens(const ov::PartialShape& shape) {
    if (shape.is_dynamic()) {
        return false;
    }
    for (auto i = 0; i + 1 < shape.rank().get_length(); ++i) {
        if (shape[i].get_length() != 1) {
            return false;
        }
    }
    return true;
}

// The highest dimension past 0 that is 1 in every given shape.
std::optional<std::size_t> shared_singleton(const std::vector<ov::Shape>& shapes) {
    for (std::size_t i = shapes.front().size(); i-- > 1;) {
        const auto is_one = [i, &shapes](const ov::Shape& s) {
            return s.size() == shapes.front().size() && s[i] == 1;
        };
        if (std::all_of(shapes.begin(), shapes.end(), is_one)) {
            return i;
        }
    }
    return std::nullopt;
}

std::shared_ptr<ov::Node> reshape_to(const ov::Output<ov::Node>& source, const ov::Shape& shape) {
    auto target = ov::op::v0::Constant::create(ov::element::i64,
                                               ov::Shape{shape.size()},
                                               std::vector<int64_t>(shape.begin(), shape.end()));
    return std::make_shared<ov::op::v1::Reshape>(source, target, false);
}

// Feeds the given inputs of `op` without dimension `drop` and with a batch of 1
// in front, then restores the original output shape for its consumers.
void rebatch(const std::shared_ptr<ov::Node>& op, const std::vector<std::size_t>& inputs, std::size_t drop) {
    const auto out_shape = op->get_output_shape(0);
    const auto consumers = op->output(0).get_target_inputs();
    for (const auto idx : inputs) {
        const auto& shape = op->get_input_shape(idx);
        ov::Shape batch_one{1};
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i != drop) {
                batch_one.push_back(shape[i]);
            }
        }
        op->input(idx).replace_source_output(reshape_to(op->input_value(idx), batch_one));
    }
    op->validate_and_infer_types();
    const auto restored = reshape_to(op, out_shape);
    for (auto consumer : consumers) {
        consumer.replace_source_output(restored);
    }
}

bool has_static_output(const std::shared_ptr<ov::Node>& node) {
    return node->get_output_size() == 1 && node->get_output_partial_shape(0).is_static();
}

class RebatchLinear final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::RebatchLinear");

    explicit RebatchLinear(std::size_t tokens) {
        auto matmul_pattern = opp::wrap_type<ov::op::v0::MatMul>();
        ov::matcher_pass_callback callback = [tokens](opp::Matcher& m) {
            const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_root());
            if (!matmul || !has_static_output(matmul) || matmul->get_transpose_a() ||
                !is_token_major(matmul->get_input_partial_shape(0), tokens) ||
                matmul->get_input_partial_shape(1).rank().get_length() != 2) {
                return false;
            }
            // The output keeps the contraction dimension out of the choice.
            const auto drop = shared_singleton({matmul->get_input_shape(0), matmul->get_output_shape(0)});
            if (!drop) {
                return false;
            }
            rebatch(matmul, {0}, *drop);
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul_pattern, "RebatchLinear"), callback);
    }
};

class OuterProductToMultiply final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::OuterProductToMultiply");

    explicit OuterProductToMultiply(std::size_t tokens) {
        auto matmul_pattern = opp::wrap_type<ov::op::v0::MatMul>();
        ov::matcher_pass_callback callback = [this, tokens](opp::Matcher& m) {
            const auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_root());
            if (!matmul || !has_static_output(matmul) || matmul->get_transpose_a() || matmul->get_transpose_b()) {
                return false;
            }
            const auto& act = matmul->get_input_partial_shape(0);
            const auto& rhs = matmul->get_input_partial_shape(1);
            if (!is_token_major(act, tokens) || act.rank().get_length() != 3 || !rhs.is_static() ||
                rhs.rank().get_length() != 3 || act[2].get_length() != 1 || rhs[1].get_length() != 1) {
                return false;
            }
            auto product = std::make_shared<ov::op::v1::Multiply>(matmul->input_value(0), matmul->input_value(1));
            product->set_friendly_name(matmul->get_friendly_name());
            ov::replace_node(matmul, product);
            register_new_node(product);  // for RebatchEltwise
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(matmul_pattern, "OuterProductToMultiply"), callback);
    }
};

class RebatchEltwise final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::RebatchEltwise");

    explicit RebatchEltwise(std::size_t tokens) {
        auto eltwise_pattern =
            opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Add, ov::op::v1::Subtract, ov::op::v1::Divide>();
        ov::matcher_pass_callback callback = [tokens](opp::Matcher& m) {
            const auto op = m.get_match_root();
            if (!has_static_output(op)) {
                return false;
            }
            std::vector<std::size_t> token_inputs;
            std::vector<ov::Shape> token_shapes;
            for (std::size_t i = 0; i < op->get_input_size(); ++i) {
                const auto& shape = op->get_input_partial_shape(i);
                if (is_token_major(shape, tokens)) {
                    token_inputs.push_back(i);
                    token_shapes.push_back(op->get_input_shape(i));
                } else if (!broadcasts_over_tokens(shape)) {
                    return false;
                }
            }
            if (token_inputs.empty()) {
                return false;
            }
            // The dropped dimension must be 1 in the output too, not one an operand broadcasts into.
            token_shapes.push_back(op->get_output_shape(0));
            const auto drop = shared_singleton(token_shapes);
            if (!drop) {
                return false;
            }
            rebatch(op, token_inputs, *drop);
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(eltwise_pattern, "RebatchEltwise"), callback);
    }
};

}  // namespace

ov::npuw::RelayoutTokenMajor::RelayoutTokenMajor(std::size_t tokens) {
    if (tokens <= 1) {
        return;
    }
    add_matcher<RebatchLinear>(tokens);
    add_matcher<OuterProductToMultiply>(tokens);
    add_matcher<RebatchEltwise>(tokens);
}
