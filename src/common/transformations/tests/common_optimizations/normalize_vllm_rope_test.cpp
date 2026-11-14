// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/normalize_vllm_rope.hpp"

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

using View = std::function<Output<Node>(const Output<Node>&)>;

struct RoPEParams {
    std::string name;
    PartialShape x_shape;
    PartialShape cos_shape;
    PartialShape sin_shape;
    std::vector<int64_t> split_lengths;  // empty: Split(num_splits=2); else VariadicSplit
    int64_t concat_axis = -1;
    bool lengths_as_input = false;
    View view = nullptr;  // applied to both halves before the Multiplies
};

std::shared_ptr<v0::Parameter> param(const PartialShape& shape, element::Type type = element::f32) {
    return std::make_shared<v0::Parameter>(type, shape);
}

// vLLM's lowering: Concat(x1 * cos + (x2 * sin) * -1, x2 * cos + x1 * sin).
Output<Node> vllm_rope(const Output<Node>& x,
                       const Output<Node>& cos,
                       const Output<Node>& sin,
                       const RoPEParams& p,
                       ParameterVector& params) {
    auto axis = v0::Constant::create(element::i64, Shape{}, {-1});
    std::shared_ptr<Node> split;
    if (p.split_lengths.empty()) {
        split = std::make_shared<v1::Split>(x, axis, 2);
    } else {
        std::shared_ptr<Node> lengths =
            v0::Constant::create(element::i64, Shape{p.split_lengths.size()}, p.split_lengths);
        if (p.lengths_as_input) {
            params.push_back(param(Shape{2}, element::i64));
            lengths = params.back();
        }
        split = std::make_shared<v1::VariadicSplit>(x, axis, lengths);
    }
    auto x1 = p.view ? p.view(split->output(0)) : split->output(0);
    auto x2 = p.view ? p.view(split->output(1)) : split->output(1);
    auto minus_one = v0::Constant::create(element::f32, Shape{}, {-1.0f});
    auto first =
        std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(x1, cos),
                                  std::make_shared<v1::Multiply>(std::make_shared<v1::Multiply>(x2, sin), minus_one));
    auto second =
        std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(x2, cos), std::make_shared<v1::Multiply>(x1, sin));
    return std::make_shared<v0::Concat>(OutputVector{first, second}, p.concat_axis);
}

// The rotate-half form NormalizeVLLMRoPE emits, on the given full-width tables.
Output<Node> rotate_half_rope(const Output<Node>& x, const Output<Node>& cos_full, const Output<Node>& sin_full) {
    const auto last = static_cast<int64_t>(x.get_partial_shape().size()) - 1;
    const auto half = x.get_partial_shape()[last].get_length() / 2;
    auto split = std::make_shared<v1::VariadicSplit>(x,
                                                     v0::Constant::create(element::i64, Shape{}, {last}),
                                                     v0::Constant::create(element::i64, Shape{2}, {half, half}));
    auto neg_x2 = std::make_shared<v1::Multiply>(split->output(1), v0::Constant::create(element::f32, Shape{}, {-1}));
    auto rotated = std::make_shared<v0::Concat>(OutputVector{neg_x2, split->output(0)}, -1);
    return std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(x, cos_full),
                                     std::make_shared<v1::Multiply>(rotated, sin_full));
}

std::shared_ptr<Model> getModel(const RoPEParams& p) {
    ParameterVector params{param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
    auto rope = vllm_rope(params[0], params[1], params[2], p, params);
    return std::make_shared<Model>(OutputVector{rope}, params);
}

std::shared_ptr<Model> getModelRef(const RoPEParams& p) {
    ParameterVector params{param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
    auto cos_full = std::make_shared<v0::Concat>(OutputVector{params[1], params[1]}, -1);
    auto sin_full = std::make_shared<v0::Concat>(OutputVector{params[2], params[2]}, -1);
    return std::make_shared<Model>(OutputVector{rotate_half_rope(params[0], cos_full, sin_full)}, params);
}

Output<Node> squeeze_axis1(const Output<Node>& o) {
    return std::make_shared<v0::Squeeze>(o, v0::Constant::create(element::i64, Shape{1}, {1}));
}

Output<Node> unsqueeze_axis1(const Output<Node>& o) {
    return std::make_shared<v0::Unsqueeze>(o, v0::Constant::create(element::i64, Shape{1}, {1}));
}

class NormalizeVLLMRoPETest : public TransformationTestsF, public testing::WithParamInterface<RoPEParams> {
public:
    NormalizeVLLMRoPETest() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<RoPEParams>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
    }
};

class NormalizeVLLMRoPEFusedTest : public NormalizeVLLMRoPETest {};
class NormalizeVLLMRoPENotFusedTest : public NormalizeVLLMRoPETest {};

}  // namespace

TEST_P(NormalizeVLLMRoPEFusedTest, Fused) {
    model = getModel(GetParam());
    model_ref = getModelRef(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    NormalizeVLLMRoPEFusedTest,
    testing::Values(
        RoPEParams{"split", Shape{1, 4, 8}, Shape{1, 4, 4}, Shape{1, 4, 4}, {}},
        RoPEParams{"variadic_split", Shape{1, 4, 8}, Shape{1, 4, 4}, Shape{1, 4, 4}, {4, 4}},
        RoPEParams{"variadic_split_inferred_length", Shape{1, 4, 8}, Shape{1, 4, 4}, Shape{1, 4, 4}, {4, -1}},
        RoPEParams{"dynamic_tokens", PartialShape{-1, 4, 8}, PartialShape{-1, 1, 4}, PartialShape{-1, 1, 4}, {}},
        RoPEParams{"positive_concat_axis", Shape{5, 4, 8}, Shape{5, 1, 4}, Shape{5, 1, 4}, {}, 2}),
    NormalizeVLLMRoPETest::getTestCaseName);

TEST_P(NormalizeVLLMRoPENotFusedTest, NotFused) {
    model = getModel(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    NormalizeVLLMRoPENotFusedTest,
    testing::Values(
        // [7, 1] halves are not neox RoPE; rewriting them as [D/2, D/2] would change the output shape.
        RoPEParams{"unequal_split", Shape{5, 4, 8}, Shape{}, Shape{}, {7, 1}},
        // Outputs 0 and 1 of a [4, 4, 4] split look like halves, but the input is 12 wide.
        RoPEParams{"three_way_split", Shape{5, 4, 12}, Shape{5, 4, 4}, Shape{5, 4, 4}, {4, 4, 4}},
        RoPEParams{"non_constant_split_lengths", Shape{5, 4, 8}, Shape{5, 4, 4}, Shape{5, 4, 4}, {4, 4}, -1, true},
        RoPEParams{"non_last_concat_axis", Shape{5, 4, 8}, Shape{5, 1, 4}, Shape{5, 1, 4}, {}, 0},
        RoPEParams{"dynamic_cos_width", Shape{5, 4, 8}, PartialShape{5, 1, -1}, Shape{5, 1, 4}, {}},
        // The CPU RoPE fusion would treat a broadcast sin as a full table.
        RoPEParams{"sin_shape_differs_from_cos", Shape{5, 4, 8}, Shape{5, 1, 4}, Shape{5, 1, 1}, {}},
        RoPEParams{"sin_broadcasts_tokens", Shape{5, 4, 8}, Shape{5, 1, 4}, Shape{1, 1, 4}, {}},
        RoPEParams{"broadcast_cos_sin", Shape{5, 4, 8}, Shape{5, 1, 1}, Shape{5, 1, 1}, {}},
        // A view on the halves changes which axis they broadcast on against cos/sin.
        RoPEParams{"squeezed_halves", Shape{2, 1, 4}, Shape{2, 2, 2}, Shape{2, 2, 2}, {}, -1, false, squeeze_axis1},
        RoPEParams{"unsqueezed_halves", Shape{3, 4}, Shape{3, 5, 2}, Shape{3, 5, 2}, {}, -1, false, unsqueeze_axis1}),
    NormalizeVLLMRoPETest::getTestCaseName);

// q and k on one cos/sin: each site is rewritten, with its own full-width tables.
TEST_F(TransformationTestsF, NormalizeVLLMRoPETwoSitesSharingCosSin) {
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    const RoPEParams p{"shared", Shape{1, 4, 8}, Shape{1, 4, 4}, Shape{1, 4, 4}, {}};
    {
        ParameterVector params{param(p.x_shape), param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
        auto q = vllm_rope(params[0], params[2], params[3], p, params);
        auto k = vllm_rope(params[1], params[2], params[3], p, params);
        model = std::make_shared<Model>(OutputVector{q, k}, params);
    }
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
    {
        ParameterVector params{param(p.x_shape), param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
        OutputVector ropes;
        for (const auto& x : {params[0], params[1]}) {
            auto cos_full = std::make_shared<v0::Concat>(OutputVector{params[2], params[2]}, -1);
            auto sin_full = std::make_shared<v0::Concat>(OutputVector{params[3], params[3]}, -1);
            ropes.push_back(rotate_half_rope(x, cos_full, sin_full));
        }
        model_ref = std::make_shared<Model>(ropes, params);
    }
}

// CommonOptimizations follows the pass with SharedOpOptimization, which merges the per-site tables back to one.
TEST_F(TransformationTestsF, NormalizeVLLMRoPEWithSharedOpOptimizationSharesTables) {
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    const RoPEParams p{"shared", Shape{1, 4, 8}, Shape{1, 4, 4}, Shape{1, 4, 4}, {}};
    {
        ParameterVector params{param(p.x_shape), param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
        auto q = vllm_rope(params[0], params[2], params[3], p, params);
        auto k = vllm_rope(params[1], params[2], params[3], p, params);
        model = std::make_shared<Model>(OutputVector{q, k}, params);
    }
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
    manager.register_pass<ov::pass::SharedOpOptimization>();
    {
        ParameterVector params{param(p.x_shape), param(p.x_shape), param(p.cos_shape), param(p.sin_shape)};
        auto cos_full = std::make_shared<v0::Concat>(OutputVector{params[2], params[2]}, -1);
        auto sin_full = std::make_shared<v0::Concat>(OutputVector{params[3], params[3]}, -1);
        model_ref = std::make_shared<Model>(OutputVector{rotate_half_rope(params[0], cos_full, sin_full),
                                                         rotate_half_rope(params[1], cos_full, sin_full)},
                                            params);
    }
}

TEST_F(TransformationTestsF, NormalizeVLLMRoPENotFusedOnUnrelatedConcat) {
    ParameterVector params{param(Shape{1, 4, 8}), param(Shape{1, 4, 8}), param(Shape{1, 4, 8})};
    model = std::make_shared<Model>(
        OutputVector{std::make_shared<v0::Concat>(OutputVector{params[0], params[1], params[2]}, -1)},
        params);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}
