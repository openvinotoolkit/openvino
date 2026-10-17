// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/cpu_opset/common/op/ngram_hash.hpp"
#include "transformations/cpu_opset/common/pass/ngram_hash_fusion.hpp"

using namespace ov;

namespace {

// Builds: offsets + floor_mod(unsqueeze(xor(t0 * m0, t1 * m1), -1), moduli)
// i.e. the 2-term (bigram) shape of the int64 n-gram hash this pass targets.
std::shared_ptr<Model> build_model(element::Type token_type, int64_t m0, int64_t m1,
                                    const std::vector<int64_t>& moduli, const std::vector<int64_t>& offsets,
                                    const Shape& token_shape) {
    auto t0 = std::make_shared<op::v0::Parameter>(token_type, token_shape);
    auto t1 = std::make_shared<op::v0::Parameter>(token_type, token_shape);

    auto m0_const = op::v0::Constant::create(token_type, Shape{}, {m0});
    auto m1_const = op::v0::Constant::create(token_type, Shape{}, {m1});
    auto mul0 = std::make_shared<op::v1::Multiply>(t0, m0_const);
    auto mul1 = std::make_shared<op::v1::Multiply>(t1, m1_const);
    auto mixed = std::make_shared<op::v13::BitwiseXor>(mul0, mul1);

    auto last_axis = op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto mixed_unsq = std::make_shared<op::v0::Unsqueeze>(mixed, last_axis);

    auto moduli_const = op::v0::Constant::create(token_type, Shape{moduli.size()}, moduli);
    auto floor_mod = std::make_shared<op::v1::FloorMod>(mixed_unsq, moduli_const);

    auto offsets_const = op::v0::Constant::create(token_type, Shape{offsets.size()}, offsets);
    auto result = std::make_shared<op::v1::Add>(floor_mod, offsets_const);

    return std::make_shared<Model>(OutputVector{result}, ParameterVector{t0, t1});
}

bool has_floor_mod(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<op::v1::FloorMod>(node)) {
            return true;
        }
    }
    return false;
}

bool has_ngram_hash(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::intel_cpu::NgramHashNode>(node)) {
            return true;
        }
    }
    return false;
}

int64_t floor_mod_ref(int64_t a, int64_t m) {
    int64_t r = a % m;
    if (r < 0) {
        r += m;
    }
    return r;
}

// int64 multiply with two's-complement wrap-around, as torch/the traced graph does.
int64_t wrap_mul(int64_t a, int64_t b) {
    return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

void check_on_cpu(int64_t m0, int64_t m1, const std::vector<int64_t>& moduli, const std::vector<int64_t>& offsets) {
    const Shape token_shape{2, 5};
    auto model = build_model(element::i64, m0, m1, moduli, offsets, token_shape);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::NgramHashFusion>();
    manager.run_passes(model);
    EXPECT_FALSE(has_floor_mod(model)) << "pass should have rewritten the FloorMod/i64-Multiply subtree";
    EXPECT_TRUE(has_ngram_hash(model));

    Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto request = compiled.create_infer_request();

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, 300000);
    size_t count = shape_size(token_shape);
    std::vector<int64_t> t0_data(count), t1_data(count);
    for (size_t i = 0; i < count; ++i) {
        t0_data[i] = dist(rng);
        t1_data[i] = dist(rng);
    }
    Tensor t0_tensor(element::i64, token_shape, t0_data.data());
    Tensor t1_tensor(element::i64, token_shape, t1_data.data());
    request.set_input_tensor(0, t0_tensor);
    request.set_input_tensor(1, t1_tensor);
    request.infer();

    auto output = request.get_output_tensor();
    ASSERT_EQ(output.get_element_type(), element::i64);
    const auto* actual = output.data<int64_t>();

    for (size_t i = 0; i < count; ++i) {
        int64_t mixed = wrap_mul(t0_data[i], m0) ^ wrap_mul(t1_data[i], m1);
        for (size_t h = 0; h < moduli.size(); ++h) {
            int64_t expected = floor_mod_ref(mixed, moduli[h]) + offsets[h];
            EXPECT_EQ(expected, actual[i * moduli.size() + h]) << "token_idx=" << i << " head=" << h;
        }
    }
}

}  // namespace

// The real bug this pass fixes: CPU's ConvertPrecision unconditionally downcasts every i64 op to
// i32 (see transformation_pipeline.cpp's i64->i32 entry), silently truncating this hash's
// intermediate products/XORs regardless of whether any constant itself overflows int32. So the
// correctness check here must go through the actual CPU plugin (statically linked into this test
// binary): ov::Model::evaluate() has no reference kernel for the fused NgramHash node.
TEST(NgramHashFusionTest, MatchesReferenceOnCpu) {
    check_on_cpu(1234567891011LL,  // does not fit int32
                 777013,           // fits int32, but token*m1 still doesn't
                 {1000003, 1000033},
                 {0, 1000000});
}

TEST(NgramHashFusionTest, WrapsAroundLikeInt64) {
    // token * multiplier overflows int64, so mixed values go negative: checks wrap-around + floor_mod sign.
    check_on_cpu(static_cast<int64_t>(0x9E3779B97F4A7C15ULL),
                 0x3BF58476D1CE4E5BLL,
                 {20000003, 20000023, 20000033},
                 {0, 20000003, 40000026});
}

TEST(NgramHashFusionTest, DoesNotFireOnI32) {
    const Shape token_shape{2, 3};
    const int64_t m0 = 777013;
    const int64_t m1 = 12345;
    const std::vector<int64_t> moduli{1000003, 1000033};
    const std::vector<int64_t> offsets{0, 1000000};

    auto model = build_model(element::i32, m0, m1, moduli, offsets, token_shape);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::NgramHashFusion>();
    manager.run_passes(model);

    EXPECT_TRUE(has_floor_mod(model)) << "pass should be a no-op for i32 (CPU doesn't downcast i32 further)";
}

TEST(NgramHashFusionTest, DoesNotFireWhenIdsOverflowInt32) {
    auto model = build_model(element::i64, 1234567891011LL, 777013, {1000003, 1000033}, {0, int64_t{1} << 31}, Shape{2, 3});
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::NgramHashFusion>();
    manager.run_passes(model);

    EXPECT_FALSE(has_ngram_hash(model));
}
