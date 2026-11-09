// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/canonicalize_lm_head_dq_layout.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
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
// (the zero point and scale may also be per-tensor, i.e. [1]) and the expected result is the
// optimum-intel layout:
//
//   W u8[OC,IC] -> Convert -+
//   Z u8[OC,1]  -> Convert -+-> Subtract -> Multiply(S f16[OC,1]) -> MatMul(transpose_b=true)
//
// All constants hold non-uniform data, so a wrong reshape / transpose order changes the result and is
// caught by the structural comparison (CONST_VALUES) as well as by the accuracy check.

namespace {

using namespace ov;
using namespace ov::op;

constexpr size_t OC = 64;  // stands in for the 151936-wide vocab
constexpr size_t IC = 32;  // stands in for hidden size 2560

std::shared_ptr<v0::Constant> make_u8(const Shape& shape, size_t mul, size_t add) {
    std::vector<uint8_t> data(shape_size(shape));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * mul + add) % 251);
    }
    return std::make_shared<v0::Constant>(element::u8, shape, data);
}

std::shared_ptr<v0::Constant> make_scale(const Shape& shape) {
    std::vector<float> data(shape_size(shape));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = 0.01f * static_cast<float>(1 + i % 17);
    }
    return std::make_shared<v0::Constant>(element::f16, shape, data);
}

std::shared_ptr<Node> make_reshape(const Output<Node>& in, const std::vector<int64_t>& shape) {
    return std::make_shared<v1::Reshape>(in, v0::Constant::create(element::i64, Shape{shape.size()}, shape), true);
}

struct ChainOptions {
    element::Type weight_type = element::u8;
    size_t out_channels = OC;
    size_t zerop_size = OC;  // OC - per-channel, 1 - per-tensor
    size_t scale_size = OC;
    bool reshape_zerop_before_convert = false;
    std::vector<int64_t> transpose_perm = {1, 0};
    bool transpose_a = false;
    bool extra_weight_consumer = false;  // the source weight Constant also feeds a second Result
};

std::shared_ptr<Model> getModel(const ChainOptions& opt = {}) {
    const size_t oc = opt.out_channels;
    std::shared_ptr<v0::Constant> weight = make_u8(Shape{oc, 1, IC}, 7, 3);
    if (opt.weight_type != element::u8) {
        weight = std::make_shared<v0::Constant>(opt.weight_type, Shape{oc, 1, IC}, 3);
    }
    auto zerop = make_u8(Shape{opt.zerop_size}, 13, 5);
    auto scale = make_scale(Shape{opt.scale_size});

    auto cvt_w = std::make_shared<v0::Convert>(weight, element::f16);
    std::shared_ptr<Node> zerop_f16;
    if (opt.reshape_zerop_before_convert) {
        zerop_f16 = std::make_shared<v0::Convert>(make_reshape(zerop, {0, 1, 1}), element::f16);
    } else {
        zerop_f16 = make_reshape(std::make_shared<v0::Convert>(zerop, element::f16), {0, 1, 1});
    }
    auto sub = std::make_shared<v1::Subtract>(cvt_w, zerop_f16);
    auto mul = std::make_shared<v1::Multiply>(sub, make_reshape(scale, {0, 1, 1}));
    auto weight_2d = make_reshape(mul, {static_cast<int64_t>(oc), static_cast<int64_t>(IC)});
    auto perm = v0::Constant::create(element::i64, Shape{2}, opt.transpose_perm);
    auto transpose = std::make_shared<v1::Transpose>(weight_2d, perm);

    auto act = std::make_shared<v0::Parameter>(element::f16, opt.transpose_a ? Shape{1, IC, 1} : Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, transpose, opt.transpose_a, false);

    ResultVector results{std::make_shared<v0::Result>(matmul)};
    if (opt.extra_weight_consumer) {
        results.push_back(std::make_shared<v0::Result>(std::make_shared<v0::Convert>(weight, element::f16)));
    }
    return std::make_shared<Model>(results, ParameterVector{act});
}

std::shared_ptr<Model> getModelRef(size_t zerop_size = OC, size_t scale_size = OC, bool extra_weight_consumer = false) {
    auto weight = make_u8(Shape{OC, IC}, 7, 3);
    auto zerop = make_u8(Shape{zerop_size, 1}, 13, 5);
    auto scale = make_scale(Shape{scale_size, 1});

    auto cvt_w = std::make_shared<v0::Convert>(weight, element::f16);
    auto cvt_z = std::make_shared<v0::Convert>(zerop, element::f16);
    auto sub = std::make_shared<v1::Subtract>(cvt_w, cvt_z);
    auto mul = std::make_shared<v1::Multiply>(sub, scale);

    auto act = std::make_shared<v0::Parameter>(element::f16, Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, mul, false, true);

    ResultVector results{std::make_shared<v0::Result>(matmul)};
    if (extra_weight_consumer) {
        // The untouched consumer keeps reading the original [OC,1,IC] Constant.
        results.push_back(std::make_shared<v0::Result>(
            std::make_shared<v0::Convert>(make_u8(Shape{OC, 1, IC}, 7, 3), element::f16)));
    }
    return std::make_shared<Model>(results, ParameterVector{act});
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
    comparator.enable(FunctionsComparator::ACCURACY);
    model = getModel();
    model_ref = getModelRef();
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, ZeroPointReshapedBeforeConvert) {
    comparator.enable(FunctionsComparator::ACCURACY);
    ChainOptions opt;
    opt.reshape_zerop_before_convert = true;
    model = getModel(opt);
    model_ref = getModelRef();
}

// A per-tensor zero point / scale broadcasts against [OC,1,IC] the same way after the rewrite.
TEST_F(CanonicalizeLMHeadDQLayoutTest, PerTensorZeroPoint) {
    comparator.enable(FunctionsComparator::ACCURACY);
    ChainOptions opt;
    opt.zerop_size = 1;
    model = getModel(opt);
    model_ref = getModelRef(/*zerop_size=*/1, OC);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, PerTensorScale) {
    comparator.enable(FunctionsComparator::ACCURACY);
    ChainOptions opt;
    opt.scale_size = 1;
    model = getModel(opt);
    model_ref = getModelRef(OC, /*scale_size=*/1);
}

// The source weight Constant may have other consumers (e.g. a tied embedding): they must keep reading
// the original [OC,1,IC] weight, while the LM head gets the canonical view of the same data.
TEST_F(CanonicalizeLMHeadDQLayoutTest, SourceWeightConstantHasAnotherConsumer) {
    ChainOptions opt;
    opt.extra_weight_consumer = true;
    model = getModel(opt);
    model_ref = getModelRef(OC, OC, /*extra_weight_consumer=*/true);
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

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_TransposeOrderIsNotSwap) {
    // OC == IC makes the identity permutation a valid graph: the weight is then used as [OC,IC] with
    // transpose_b=false, which is not the layout the rewrite assumes.
    ChainOptions opt;
    opt.out_channels = IC;
    opt.zerop_size = IC;
    opt.scale_size = IC;
    opt.transpose_perm = {0, 1};
    model = getModel(opt);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_MatMulTransposeA) {
    ChainOptions opt;
    opt.transpose_a = true;
    model = getModel(opt);
}

TEST_F(CanonicalizeLMHeadDQLayoutTest, Negative_PlainMatMul) {
    auto weight = std::make_shared<v0::Constant>(element::f16, Shape{IC, OC}, 1.0f);
    auto act = std::make_shared<v0::Parameter>(element::f16, Shape{1, 1, IC});
    auto matmul = std::make_shared<v0::MatMul>(act, weight, false, false);
    model = std::make_shared<Model>(matmul, ParameterVector{act});
}

// The new Constants are views of the original data (the vocab weight is not copied) and keep the
// original runtime info. NPUW references weights by their file offset (WeightlessCacheAttribute), which
// copy_runtime_info does not copy (not copyable); without it the whole vocab weight is copied.
TEST(CanonicalizeLMHeadDQLayoutConstants, DataAndRuntimeInfoAreKept) {
    constexpr const char* custom_key = "canonicalize_lm_head_dq_layout_test_key";
    auto model = getModel();

    // original data pointer -> bin_offset
    std::unordered_map<const void*, size_t> offsets;
    size_t next_offset = 1000;
    for (const auto& op : model->get_ordered_ops()) {
        auto cnst = ov::as_type_ptr<v0::Constant>(op);
        if (!cnst || cnst->get_element_type() == element::i64) {
            continue;  // skip the Reshape / Transpose constants
        }
        auto& rt_info = cnst->get_rt_info();
        rt_info[WeightlessCacheAttribute::get_type_info_static()] =
            WeightlessCacheAttribute(cnst->get_byte_size(), next_offset, cnst->get_element_type());
        rt_info[custom_key] = std::string("kept");
        offsets[cnst->get_data_ptr()] = next_offset;
        next_offset += 1000;
    }
    ASSERT_EQ(offsets.size(), 3u);

    pass::Manager manager;
    manager.register_pass<npuw::CanonicalizeLMHeadDQLayout>();
    manager.run_passes(model);

    size_t checked = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto cnst = ov::as_type_ptr<v0::Constant>(op);
        if (!cnst) {
            continue;
        }
        const auto it_offset = offsets.find(cnst->get_data_ptr());
        ASSERT_TRUE(it_offset != offsets.end()) << cnst->get_friendly_name() << " does not share the original data";

        const auto& rt_info = cnst->get_rt_info();
        const auto it_attr = rt_info.find(WeightlessCacheAttribute::get_type_info_static());
        ASSERT_TRUE(it_attr != rt_info.end()) << cnst->get_friendly_name();
        EXPECT_EQ(it_attr->second.as<WeightlessCacheAttribute>().bin_offset, it_offset->second);

        const auto it_custom = rt_info.find(custom_key);
        ASSERT_TRUE(it_custom != rt_info.end()) << cnst->get_friendly_name();
        EXPECT_EQ(it_custom->second.as<std::string>(), "kept");
        ++checked;
    }
    EXPECT_EQ(checked, 3u);
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
