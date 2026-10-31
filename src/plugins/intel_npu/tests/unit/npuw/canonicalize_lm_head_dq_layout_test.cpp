// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/canonicalize_lm_head_dq_layout.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

// Tests for ov::npuw::CanonicalizeLMHeadDQLayout.
//
// The input graph is the LM-head chain of the WebNN-exported Qwen3-4B (sizes are reduced):
//
//   W u8[OC,1,IC] -> Convert ---------------------------------+
//   Z u8[OC] -> Convert -> Reshape[OC,1,1]  (or Reshape first) -+-> Subtract
//   S f16[OC] -> Reshape[OC,1,1] -------------------------------> Multiply
//     -> Reshape[OC,IC] -> Transpose{1,0} -> MatMul(transpose_b=false)
//
// and the expected result is the optimum-intel layout:
//
//   W u8[OC,IC] -> Convert -+
//   Z u8[OC,1]  -> Convert -+-> Subtract -> Multiply(S f16[OC,1]) -> MatMul(transpose_b=true)

namespace {

using namespace ov;
using namespace ov::op;

constexpr size_t OC = 64;  // stands in for the 151936-wide vocab
constexpr size_t IC = 32;  // stands in for hidden size 2560

std::shared_ptr<Node> make_reshape(const Output<Node>& in, const std::vector<int64_t>& shape) {
    return std::make_shared<v1::Reshape>(in, v0::Constant::create(element::i64, Shape{shape.size()}, shape), true);
}

struct ChainOptions {
    element::Type weight_type = element::u8;
    Shape zerop_shape = {OC};
    Shape scale_shape = {OC};
    bool reshape_zerop_before_convert = false;
};

std::shared_ptr<Model> getModel(const ChainOptions& opt = {}) {
    auto weight = std::make_shared<v0::Constant>(opt.weight_type, Shape{OC, 1, IC}, 3);
    auto zerop = std::make_shared<v0::Constant>(element::u8, opt.zerop_shape, 1);
    auto scale = std::make_shared<v0::Constant>(element::f16, opt.scale_shape, 0.5f);

    auto cvt_w = std::make_shared<v0::Convert>(weight, element::f16);
    std::shared_ptr<Node> zerop_f16;
    if (opt.reshape_zerop_before_convert) {
        zerop_f16 = std::make_shared<v0::Convert>(make_reshape(zerop, {0, 1, 1}), element::f16);
    } else {
        zerop_f16 = make_reshape(std::make_shared<v0::Convert>(zerop, element::f16), {0, 1, 1});
    }
    auto sub = std::make_shared<v1::Subtract>(cvt_w, zerop_f16);
    auto mul = std::make_shared<v1::Multiply>(sub, make_reshape(scale, {0, 1, 1}));
    auto weight_2d = make_reshape(mul, {static_cast<int64_t>(OC), static_cast<int64_t>(IC)});
    auto perm = v0::Constant::create(element::i64, Shape{2}, {1, 0});
    auto transpose = std::make_shared<v1::Transpose>(weight_2d, perm);

    auto act = std::make_shared<v0::Parameter>(element::f16, Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, transpose, false, false);
    return std::make_shared<Model>(matmul, ParameterVector{act});
}

std::shared_ptr<Model> getModelRef() {
    auto weight = std::make_shared<v0::Constant>(element::u8, Shape{OC, IC}, 3);
    auto zerop = std::make_shared<v0::Constant>(element::u8, Shape{OC, 1}, 1);
    auto scale = std::make_shared<v0::Constant>(element::f16, Shape{OC, 1}, 0.5f);

    auto cvt_w = std::make_shared<v0::Convert>(weight, element::f16);
    auto cvt_z = std::make_shared<v0::Convert>(zerop, element::f16);
    auto sub = std::make_shared<v1::Subtract>(cvt_w, cvt_z);
    auto mul = std::make_shared<v1::Multiply>(sub, scale);

    auto act = std::make_shared<v0::Parameter>(element::f16, Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, mul, false, true);
    return std::make_shared<Model>(matmul, ParameterVector{act});
}

class CanonicalizeLMHeadDQLayoutTest : public TransformationTestsF {
public:
    CanonicalizeLMHeadDQLayoutTest() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<npuw::CanonicalizeLMHeadDQLayout>();
    }
};

}  // namespace

TEST_F(CanonicalizeLMHeadDQLayoutTest, ZeroPointReshapedAfterConvert) {
    model = getModel();
    model_ref = getModelRef();
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, ZeroPointReshapedBeforeConvert) {
    ChainOptions opt;
    opt.reshape_zerop_before_convert = true;
    model = getModel(opt);
    model_ref = getModelRef();
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, AlreadyCanonicalLayoutIsKept) {
    model = getModelRef();
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, SecondRunKeepsGraphUnchanged) {
    model = getModel();
    model_ref = getModelRef();
    manager.register_pass<npuw::CanonicalizeLMHeadDQLayout>();
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_NonU8Weight) {
    ChainOptions opt;
    opt.weight_type = element::i8;
    model = getModel(opt);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_PerTensorZeroPoint) {
    ChainOptions opt;
    opt.zerop_shape = {1};
    model = getModel(opt);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_PerTensorScale) {
    ChainOptions opt;
    opt.scale_shape = {1};
    model = getModel(opt);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_PlainMatMul) {
    auto weight = std::make_shared<v0::Constant>(element::f16, Shape{IC, OC}, 1.0f);
    auto act = std::make_shared<v0::Parameter>(element::f16, Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, weight, false, false);
    model = std::make_shared<Model>(matmul, ParameterVector{act});
}

// The contract this pass exists for: the canonical layout is recognized by the tail-vocab matcher,
// the WebNN layout is not. This checks a property of another pass, which a model_ref comparison
// can't express, hence the explicit pass::Manager.
TEST(CanonicalizeLMHeadDQLayoutIntegration, PreserveConstDictMatMulAsymmMatchesOnlyAfterPass) {
    using ConstVector = std::vector<std::shared_ptr<v0::Constant>>;
    const auto count_kept = [](const std::shared_ptr<Model>& m) {
        ConstVector to_keep;
        npuw::patterns::opt::Context ctx(m, m->get_parameters().size());
        ctx.mm_gate = false;
        pass::Manager manager;
        manager.register_pass<npuw::patterns::opt::PreserveConstDictMatMulAsymm>(std::ref(ctx), std::ref(to_keep));
        manager.run_passes(m);
        return to_keep.size();
    };

    auto model = getModel();
    EXPECT_EQ(count_kept(model), 0u);

    pass::Manager manager;
    manager.register_pass<npuw::CanonicalizeLMHeadDQLayout>();
    manager.run_passes(model);
    EXPECT_EQ(count_kept(model), 3u);  // weight, zero point, scale
}
