// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_grouped_matmul_to_compressed.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

using namespace testing;
using namespace ov::pass;

namespace {

using ov::test::utils::DecompressionType;

struct ConvertGMMToCompressedParams {
    ov::element::Type weights_type;
    int group_size;  // -1 for per-channel
    size_t K;
    size_t N;
    bool has_transpose;  // true: the Transpose node is kept on the weights branch
    bool with_offsets;   // 2D x 3D form when true, 3D x 3D otherwise
    bool dynamic_m;      // dynamic row count of mat_a (3D x 3D form only)
    DecompressionType multiply_type;
    DecompressionType subtract_type;
    bool reshape_on_decompression_constant;
    bool expect_match = true;  // false: the pass is expected to leave the GroupedMatMul untouched
};

constexpr size_t G_EXPERTS = 4;

// Weights, scales and zero points are built by initMatMulDecompressionSubgraph, so scalar / empty
// scale and zero point, the Transpose branch, grouped [G, N, K/gs, gs] layouts and reshaped
// decompression constants are covered, as well as the G computation for each of them.
class ConvertGMMToCompressed : public testing::WithParamInterface<ConvertGMMToCompressedParams>, public testing::Test {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<ConvertGMMToCompressedParams>& obj) {
        const auto& p = obj.param;
        auto dt_name = [](DecompressionType t) {
            switch (t) {
            case DecompressionType::full:
                return "full";
            case DecompressionType::scalar:
                return "scalar";
            case DecompressionType::empty:
                return "empty";
            }
            return "unknown";
        };
        std::ostringstream result;
        result << "weights_type=" << p.weights_type << "_";
        result << "group_size=" << (p.group_size < 0 ? std::string("per_channel") : std::to_string(p.group_size))
               << "_";
        result << "K=" << p.K << "_N=" << p.N << "_";
        result << "transpose=" << p.has_transpose << "_";
        result << "with_offsets=" << p.with_offsets << "_";
        result << "dynamic_m=" << p.dynamic_m << "_";
        result << "multiply=" << dt_name(p.multiply_type) << "_";
        result << "subtract=" << dt_name(p.subtract_type) << "_";
        result << "reshape_const=" << p.reshape_on_decompression_constant << "_";
        result << "expect_match=" << p.expect_match << "_";
        result << "idx=" << obj.index;
        return result.str();
    }
};

std::shared_ptr<ov::op::v0::Constant> fold_to_constant(const ov::Output<ov::Node>& value) {
    return ov::util::get_constant_from_source(value);
}

bool same_bytes(const std::shared_ptr<ov::op::v0::Constant>& a, const std::shared_ptr<ov::op::v0::Constant>& b) {
    return a && b && a->get_element_type() == b->get_element_type() && a->get_byte_size() == b->get_byte_size() &&
           std::memcmp(a->get_data_ptr(), b->get_data_ptr(), a->get_byte_size()) == 0;
}

TEST_P(ConvertGMMToCompressed, StructureValuesAndGroups) {
    const auto& p = GetParam();
    const size_t K = p.K, N = p.N;

    // Planar [G, K, N] is passed to the builder. Without Transpose the stored weights are
    // [G, N, K]; with it they are stored planar and transposed back to mat_b = [G, N, K].
    auto weights_subgraph =
        ov::test::utils::initMatMulDecompressionSubgraph(ov::Shape{G_EXPERTS, K, N},
                                                                   p.group_size,
                                                                   ov::element::f32,
                                                                   p.weights_type,
                                                                   ov::element::f32,
                                                                   ov::element::f32,
                                                                   !p.has_transpose, /*transpose_weights=*/
                                                                   p.multiply_type,
                                                                   p.subtract_type,
                                                                   p.reshape_on_decompression_constant,
                                                                   p.has_transpose /*insert_transpose_node=*/);
    ASSERT_EQ(weights_subgraph->get_output_shape(0), (ov::Shape{G_EXPERTS, N, K}));

    ov::ParameterVector params;
    std::shared_ptr<ov::Node> gmm;
    if (p.with_offsets) {
        auto input =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(K)});
        auto offsets = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{G_EXPERTS});
        params = {input, offsets};
        gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, weights_subgraph, offsets);
    } else {
        const ov::Dimension m = p.dynamic_m ? ov::Dimension::dynamic() : ov::Dimension(10);
        auto input =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                    ov::PartialShape{ov::Dimension(G_EXPERTS), m, ov::Dimension(K)});
        params = {input};
        gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, weights_subgraph);
    }
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gmm}, params);

    // Original constants: weights are the biggest constant of weights_type, zero point is the other
    // one of that type, scale is the f32 one.
    std::shared_ptr<ov::op::v0::Constant> src_wei, src_zp, src_scale;
    for (const auto& op : model->get_ops()) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!c) {
            continue;
        }
        if (c->get_element_type() == p.weights_type) {
            if (!src_wei || ov::shape_size(c->get_shape()) > ov::shape_size(src_wei->get_shape())) {
                src_zp = src_wei;
                src_wei = c;
            } else {
                src_zp = c;
            }
        } else if (c->get_element_type() == ov::element::f32) {
            src_scale = c;
        }
    }
    ASSERT_NE(src_wei, nullptr);

    const bool with_zp = p.subtract_type != DecompressionType::empty;
    const bool with_scale = p.multiply_type != DecompressionType::empty;

    size_t seen_g = 0;
    bool predicate_called = false;
    ov::pass::Manager manager;
    manager.register_pass<ConvertGroupedMatMulToGroupedMatMulCompressed>(
        std::vector<ov::element::Type>{p.weights_type},
        [&](const std::shared_ptr<ov::op::internal::GroupedMatMulCompressed>&, size_t, size_t, size_t g) {
            predicate_called = true;
            seen_g = g;
            return true;
        });
    manager.run_passes(model);

    if (!p.expect_match) {
        EXPECT_FALSE(predicate_called);
        for (const auto& op : model->get_ops()) {
            EXPECT_EQ(ov::as_type_ptr<ov::op::internal::GroupedMatMulCompressed>(op), nullptr);
        }
        return;
    }
    ASSERT_TRUE(predicate_called) << "pattern did not match";
    std::shared_ptr<ov::op::internal::GroupedMatMulCompressed> result;
    for (const auto& op : model->get_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::internal::GroupedMatMulCompressed>(op)) {
            result = c;
        }
    }
    ASSERT_NE(result, nullptr);

    const size_t groups = p.group_size == -1 ? 1u : K / static_cast<size_t>(p.group_size);
    EXPECT_EQ(seen_g, groups);
    EXPECT_EQ(result->get_input_size(), 3u + (p.with_offsets ? 1u : 0u) + (with_zp ? 1u : 0u));
    EXPECT_EQ(result->get_input_shape(1), (ov::Shape{G_EXPERTS, N, K}));
    EXPECT_EQ(result->get_input_element_type(1), p.weights_type);

    // Data are only kept as is when no Transpose is baked into the constants. The grouped
    // [G, N, K/gs, gs] -> [G, N, K] reshape doesn't change the byte layout.
    if (!p.has_transpose) {
        EXPECT_TRUE(same_bytes(fold_to_constant(result->input_value(1)), src_wei)) << "weights data changed";
        const size_t scale_idx = p.with_offsets ? 3 : 2;
        if (with_scale && src_scale) {
            auto res_scale = fold_to_constant(result->input_value(scale_idx));
            ASSERT_NE(res_scale, nullptr);
            EXPECT_EQ(ov::shape_size(res_scale->get_shape()), ov::shape_size(src_scale->get_shape()));
            EXPECT_TRUE(same_bytes(res_scale, src_scale)) << "scale data changed";
            if (p.multiply_type == DecompressionType::full && !p.reshape_on_decompression_constant) {
                EXPECT_EQ(ov::shape_size(res_scale->get_shape()), G_EXPERTS * N * groups);
            }
        }
        if (with_zp && src_zp) {
            auto res_zp = fold_to_constant(result->input_value(scale_idx + 1));
            ASSERT_NE(res_zp, nullptr);
            EXPECT_EQ(ov::shape_size(res_zp->get_shape()), ov::shape_size(src_zp->get_shape()));
            EXPECT_TRUE(same_bytes(res_zp, src_zp)) << "zero point data changed";
        }
    }
}

using DT = DecompressionType;
const auto params = std::vector<ConvertGMMToCompressedParams>{
    // clang-format off
    // type            gs   K     N    transp offs   dyn_m  multiply   subtract   reshape_const
    // former hand-built cases
    {ov::element::u8,  -1,  2048, 128, false, false, false, DT::full, DT::full,   false},
    {ov::element::u8,  128, 2048, 128, false, false, false, DT::full, DT::full,   false},
    {ov::element::u4,  -1,  512,  32,  false, false, true,  DT::full, DT::full,   false},
    {ov::element::u8,  -1,  2048, 128, false, true,  false, DT::full, DT::full,   false},
    {ov::element::u8,  128, 2048, 128, false, true,  false, DT::full, DT::full,   false},
    {ov::element::u4,  128, 512,  32,  false, true,  false, DT::full, DT::full,   false},
    // zero point: scalar / empty
    {ov::element::u8,  -1,  256,  64,  false, true,  false, DT::full, DT::scalar, false},
    {ov::element::u8,  -1,  256,  64,  false, true,  false, DT::full, DT::empty,  false},
    {ov::element::u4,  64,  256,  64,  false, false, false, DT::full, DT::scalar, false},
    {ov::element::u4,  64,  256,  64,  false, true,  false, DT::full, DT::empty,  false},
    {ov::element::u3,  128, 256,  64,  false, true,  false, DT::full, DT::full,   false},
    // scale: scalar / empty. A missing scale is not matched by CompressedWeightsBlock.
    {ov::element::u8,  -1,  256,  64,  false, true,  false, DT::scalar, DT::full,   false},
    {ov::element::u4,  64,  256,  64,  false, true,  false, DT::scalar, DT::full,   false},
    {ov::element::u4,  64,  256,  64,  false, false, false, DT::scalar, DT::scalar, false},
    {ov::element::u4,  64,  256,  64,  true,  true,  false, DT::scalar, DT::full,   false},
    {ov::element::u4,  64,  256,  64,  false, true,  false, DT::empty,  DT::full,   false, false},
    // Transpose kept on the weights branch
    {ov::element::u8,  -1,  256,  64,  true,  true,  false, DT::full, DT::full,   false},
    {ov::element::u8,  -1,  256,  64,  true,  false, false, DT::full, DT::empty,  false},
    {ov::element::u4,  64,  256,  64,  true,  true,  false, DT::full, DT::full,   false},
    {ov::element::u4,  64,  256,  64,  true,  true,  false, DT::full, DT::scalar, false},
    {ov::element::u4,  64,  256,  64,  true,  false, true,  DT::full, DT::empty,  false},
    // decompression constants with 1-dims removed and restored by an explicit Reshape
    {ov::element::u8,  -1,  256,  64,  false, true,  false, DT::full, DT::full,   true},
    {ov::element::u4,  64,  256,  64,  false, false, false, DT::full, DT::full,   true},
    {ov::element::u4,  64,  256,  64,  true,  true,  false, DT::full, DT::full,   true},
    {ov::element::u4,  64,  256,  64,  false, true,  false, DT::scalar, DT::scalar, true},
    {ov::element::u8,  -1,  256,  64,  false, true,  false, DT::full, DT::empty,  true},
    // clang-format on
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke,
                         ConvertGMMToCompressed,
                         ::testing::ValuesIn(params),
                         ConvertGMMToCompressed::get_test_case_name);

// A supports-predicate returning false must leave the original GroupedMatMul untouched.
TEST_F(TransformationTestsF, ConvertGMMToCompressedVetoedByPredicate) {
    const auto compressed_type = ov::element::u8;
    const ov::Shape wei_shape{4, 128, 2048};
    const ov::Shape scale_zp_shape{4, 128, 1};
    const ov::PartialShape in_shape{4, 10, 2048};

    auto build = [&]() {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
        auto weights_const = ov::op::v0::Constant::create(compressed_type, wei_shape, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, scale_zp_shape, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto gmm = std::make_shared<ov::op::v17::GroupedMatMul>(input, wei_scale);
        return std::make_shared<ov::Model>(ov::OutputVector{gmm}, ov::ParameterVector{input});
    };

    manager.register_pass<ConvertGroupedMatMulToGroupedMatMulCompressed>(
        std::vector<ov::element::Type>{compressed_type},
        [](const std::shared_ptr<ov::op::internal::GroupedMatMulCompressed>&, size_t, size_t, size_t) {
            return false;
        });

    model = build();
}
