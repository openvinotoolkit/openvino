// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/sdpa_fusion.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace testing;

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::element;

enum class InputType : int { Q, K, V, SDPA };

class SDPA {
public:
    SDPA(element::Type type, const PartialShape& q_shape, const PartialShape& k_shape, const PartialShape& v_shape)
        : m_type(type) {
        params.push_back(make_shared<op::v0::Parameter>(m_type, q_shape));
        params.push_back(make_shared<op::v0::Parameter>(m_type, k_shape));
        params.push_back(make_shared<op::v0::Parameter>(m_type, v_shape));

        nodes[InputType::Q] = params[0];
        nodes[InputType::K] = params[1];
        nodes[InputType::V] = params[2];
    }

    void set_mask(const PartialShape& new_mask_pshape) {
        with_mask = true;
        m_mask = make_shared<v0::Parameter>(m_type, new_mask_pshape);
        params.push_back(m_mask);
    }

    void set_scale(float new_scale) {
        with_scale = true;
        m_scale = new_scale;
    }

    void set_preprocessing_callback(const std::function<void(unordered_map<InputType, Output<Node>>&)>& cb) {
        m_preprocessing_callback = cb;
    }

    void reshape(InputType which, const Shape& shape) {
        auto shape_const = op::v0::Constant::create(element::i64, {shape.size()}, shape);
        nodes[which] = make_shared<op::v1::Reshape>(nodes[which], shape_const, false);
    }
    void transpose(InputType which, const vector<size_t>& order) {
        auto order_const = op::v0::Constant::create(element::i64, {order.size()}, order);
        nodes[which] = make_shared<op::v1::Transpose>(nodes[which], order_const);
    }

    void reshape_q(const Shape& shape) {
        reshape(InputType::Q, shape);
    }
    void reshape_k(const Shape& shape) {
        reshape(InputType::K, shape);
    }
    void reshape_v(const Shape& shape) {
        reshape(InputType::V, shape);
    }
    void reshape_sdpa(const Shape& shape) {
        reshape(InputType::SDPA, shape);
    }
    void transpose_q(const vector<size_t>& order) {
        transpose(InputType::Q, order);
    }
    void transpose_sdpa(const vector<size_t>& order) {
        transpose(InputType::SDPA, order);
    }
    void transpose_k(const vector<size_t>& order) {
        transpose(InputType::K, order);
    }
    void transpose_v(const vector<size_t>& order) {
        transpose(InputType::V, order);
    }

    // Build pattern (MatMul -> scale -> mask -> Softmax -> MatMul)
    void create_pattern_sdpa(bool transpose_b = false, int softmax_axis = -1) {
        m_preprocessing_callback(nodes);

        auto attn_scores = make_shared<op::v0::MatMul>(nodes[InputType::Q], nodes[InputType::K], false, transpose_b);
        shared_ptr<Node> attn_scores_scaled = attn_scores;
        if (with_scale) {
            auto scale_const = op::v0::Constant::create(m_type, {}, {m_scale});
            attn_scores_scaled = make_shared<op::v1::Multiply>(attn_scores, scale_const);
        }
        shared_ptr<Node> attn_scores_with_mask = attn_scores_scaled;
        if (with_mask) {
            attn_scores_with_mask = make_shared<op::v1::Add>(attn_scores_scaled, m_mask);
        }
        auto softmax = make_shared<op::v8::Softmax>(attn_scores_with_mask, softmax_axis);
        auto output = make_shared<op::v0::MatMul>(softmax, nodes[InputType::V]);

        nodes[InputType::SDPA] = output;
    }

    // fused ScaledDotProductAttention op
    void create_reference_sdpa() {
        auto scale_const = op::v0::Constant::create(m_type, Shape{}, {m_scale});

        shared_ptr<Node> mask_input = m_mask;
        if (!with_mask) {
            mask_input = v0::Constant::create(m_type, {}, {0.f});
        } else {
            auto mask_input_ps = mask_input->get_output_partial_shape(0);
            auto mask_input_rank = mask_input_ps.size();
            if (mask_input_rank < 2) {
                // OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                auto diff = 2 - mask_input_rank;
                std::vector<int64_t> axes(diff);
                std::iota(axes.begin(), axes.end(), 0);
                auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                auto mask_unsqueeze = std::make_shared<v0::Unsqueeze>(mask_input, axes_const);
                mask_input = mask_unsqueeze;
            } else {
                std::vector<int64_t> axes;
                // -2 because OpenVINO SDPA specification requires the attention mask to have rank >= 2.
                for (size_t i = 0; i < (mask_input_rank - 2); ++i) {
                    if (mask_input_ps[i].is_static() && mask_input_ps[i].get_length() == 1) {
                        axes.push_back(i);
                    } else {
                        break;
                    }
                }
                if (!axes.empty()) {
                    auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
                    auto mask_squeeze = std::make_shared<v0::Squeeze>(mask_input, axes_const);
                    mask_input = mask_squeeze;
                }
            }
        }

        m_preprocessing_callback(nodes);

        auto query = nodes[InputType::Q];
        auto key = nodes[InputType::K];
        auto value = nodes[InputType::V];
        nodes[InputType::SDPA] =
            std::make_shared<op::v13::ScaledDotProductAttention>(query, key, value, mask_input, scale_const, causal);
    }

    shared_ptr<Model> build_model() {
        OutputVector outputs = {nodes.at(InputType::SDPA)};
        return make_shared<Model>(outputs, params);
    }

private:
    shared_ptr<v0::Parameter> m_mask;
    ParameterVector params;
    unordered_map<InputType, Output<Node>> nodes;
    std::function<void(unordered_map<InputType, Output<Node>>&)> m_preprocessing_callback =
        [](unordered_map<InputType, Output<Node>>&) {
            return;
        };

    bool with_mask = false;
    bool with_scale = false;
    bool causal = false;

    element::Type m_type = f32;

    float m_scale = 1.0f;
};

struct SDPAFusionParams {
    ov::PartialShape q_shape;
    ov::PartialShape k_shape;
    ov::PartialShape v_shape;
    ov::PartialShape mask_shape;
    float scale;
};

namespace {

vector<size_t> get_tranpose_order(size_t rank) {
    // 1d case is not supported
    // todo: insert exception/assert
    vector<size_t> order(rank);
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[order.size() - 1], order[order.size() - 2]);
    return order;
}

}  // namespace

class SDPAFusionImplicitTranspose
    : public TransformationTestsF,
      public ::testing::WithParamInterface<std::tuple<Type, bool, bool, SDPAFusionParams>> {};

// K Shape is implicitly transposed in the pattern, so we need to transpose it back in the reference model
TEST_P(SDPAFusionImplicitTranspose, SDPAFusionTest_implicit_transpose) {
    // Parametrization
    const auto& type = std::get<0>(GetParam());
    const auto& with_mask = std::get<1>(GetParam());
    const auto& with_scale = std::get<2>(GetParam());
    const auto& param = std::get<3>(GetParam());

    // Init.
    SDPA sdpa(type, param.q_shape, param.k_shape, param.v_shape);
    SDPA sdpa_ref(type, param.q_shape, param.k_shape, param.v_shape);

    // Attention mask processing.
    if (with_mask) {
        sdpa.set_mask(param.mask_shape);
        sdpa_ref.set_mask(param.mask_shape);
    }

    // Scale processing.
    if (with_scale) {
        sdpa.set_scale(param.scale);
        sdpa_ref.set_scale(param.scale);
    }

    // SDPA model.
    {
        sdpa.create_pattern_sdpa();
        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.transpose_k(get_tranpose_order(param.q_shape.size()));
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    // comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

SDPAFusionParams
implicit_transpose_4d(int B, int H, int S_q, int S_kv, int E, int Ev, std::vector<Dimension> mask_shape, float scale) {
    return SDPAFusionParams{{B, H, S_q, E},    // Q
                            {B, H, E, S_kv},   // K
                            {B, H, S_kv, Ev},  // V
                            mask_shape.empty() ? ov::PartialShape{} : PartialShape{mask_shape},
                            scale};
}

SDPAFusionParams
implicit_transpose_3d(int B, int S_q, int S_kv, int E, int Ev, std::vector<Dimension> mask_shape, float scale) {
    return SDPAFusionParams{{B, S_q, E},    // Q
                            {B, E, S_kv},   // K
                            {B, S_kv, Ev},  // V
                            mask_shape.empty() ? ov::PartialShape{} : PartialShape{mask_shape},
                            scale};
}

INSTANTIATE_TEST_SUITE_P(SDPAFusion,
                         SDPAFusionImplicitTranspose,
                         Combine(Values(f32, f16),                  // Types
                                 Values(true, false),               // Use attention_mask
                                 Values(true, false),               // Use scale
                                 Values(implicit_transpose_4d(1,    // B (batch)
                                                              32,   // H (heads)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ),
                                        implicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              128,             // S_q (query len)
                                                              128,             // S_kv (kv len)
                                                              64,              // E (embedding)
                                                              64,              // Ev (V embedding)
                                                              {32, 128, 128},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        implicit_transpose_3d(1,    // B (batch)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ))));

SDPAFusionParams
explicit_transpose_4d(int B, int H, int S_q, int S_kv, int E, int Ev, std::vector<Dimension> mask_shape, float scale) {
    return SDPAFusionParams{{B, H, S_q, E},    // Q
                            {B, H, S_kv, E},   // K
                            {B, H, S_kv, Ev},  // V
                            mask_shape.empty() ? ov::PartialShape{} : PartialShape{mask_shape},
                            scale};
}

SDPAFusionParams
explicit_transpose_3d(int B, int S_q, int S_kv, int E, int Ev, std::vector<Dimension> mask_shape, float scale) {
    return SDPAFusionParams{{B, S_q, E},    // Q
                            {B, S_kv, E},   // K
                            {B, S_kv, Ev},  // V
                            mask_shape.empty() ? ov::PartialShape{} : PartialShape{mask_shape},
                            scale};
}

class SDPAFusionTransposeInMatmul
    : public TransformationTestsF,
      public ::testing::WithParamInterface<std::tuple<Type, bool, bool, SDPAFusionParams>> {};

// Test for SDPAFusion with transpose_b = true in MatMul
TEST_P(SDPAFusionTransposeInMatmul, SDPAFusionTest_transpose_in_matmul) {
    // Parametrization
    const auto& type = std::get<0>(GetParam());
    const auto& with_mask = std::get<1>(GetParam());
    const auto& with_scale = std::get<2>(GetParam());
    const auto& param = std::get<3>(GetParam());

    // Init.
    SDPA sdpa(type, param.q_shape, param.k_shape, param.v_shape);
    SDPA sdpa_ref(type, param.q_shape, param.k_shape, param.v_shape);

    // Attention mask processing.
    if (with_mask) {
        sdpa.set_mask(param.mask_shape);
        sdpa_ref.set_mask(param.mask_shape);
    }

    // Scale processing.
    if (with_scale) {
        sdpa.set_scale(param.scale);
        sdpa_ref.set_scale(param.scale);
    }

    // SDPA model.
    {
        bool transpose_inside_matmul = true;  // transpose_b = true for matmul
        sdpa.create_pattern_sdpa(transpose_inside_matmul);
        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(SDPAFusion,
                         SDPAFusionTransposeInMatmul,
                         Combine(Values(f32, f16),                  // Types
                                 Values(true, false),               // Use attention_mask
                                 Values(true, false),               // Use scale
                                 Values(explicit_transpose_4d(1,    // B (batch)
                                                              32,   // H (heads)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ),
                                        explicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              128,             // S_q (query len)
                                                              128,             // S_kv (kv len)
                                                              64,              // E (embedding)
                                                              64,              // Ev (V embedding)
                                                              {32, 128, 128},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        explicit_transpose_4d(1,                // B (batch)
                                                              32,               // H (heads)
                                                              -1,               // S_q (query len)
                                                              -1,               // S_kv (kv len)
                                                              32,               // E (embedding)
                                                              32,               // Ev (V embedding)
                                                              {1, 32, -1, -1},  // mask_shape
                                                              0.125f            // scale
                                                              ),
                                        explicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              10,              // S_q (query len)
                                                              10,              // S_kv (kv len)
                                                              32,              // E (embedding)
                                                              32,              // Ev (V embedding)
                                                              {1, 1, 10, 10},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        explicit_transpose_4d(1,                 // B (batch)
                                                              10,                // H (heads)
                                                              1024,              // S_q (query len)
                                                              1024,              // S_kv (kv len)
                                                              64,                // E (embedding)
                                                              64,                // Ev (V embedding)
                                                              {10, 1024, 1024},  // mask_shape
                                                              0.125f             // scale
                                                              ),
                                        explicit_transpose_3d(1,    // B (batch)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ))));

class SDPAFusionSoftmaxAxis : public TransformationTestsF,
                              public ::testing::WithParamInterface<std::tuple<Type, bool, bool, SDPAFusionParams>> {};

// Test for SDPAFusion with axis = 3 in softmax
TEST_P(SDPAFusionSoftmaxAxis, SDPAFusionTest_softmax_axis) {
    // Parametrization
    const auto& [type, with_mask, with_scale, param] = GetParam();

    // Init.
    SDPA sdpa(type, param.q_shape, param.k_shape, param.v_shape);
    SDPA sdpa_ref(type, param.q_shape, param.k_shape, param.v_shape);

    // Attention mask processing.
    if (with_mask) {
        sdpa.set_mask(param.mask_shape);
        sdpa_ref.set_mask(param.mask_shape);
    }

    // Scale processing.
    if (with_scale) {
        sdpa.set_scale(param.scale);
        sdpa_ref.set_scale(param.scale);
    }

    // SDPA model.
    {
        bool transpose_inside_matmul = true;                   // transpose_b = true for matmul
        sdpa.create_pattern_sdpa(transpose_inside_matmul, 3);  // axis = (rank size -1) (4d, so axis = 3)
        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(SDPAFusion,
                         SDPAFusionSoftmaxAxis,
                         Combine(Values(f32, f16),                  // Types
                                 Values(true, false),               // Use attention_mask
                                 Values(true, false),               // Use scale
                                 Values(explicit_transpose_4d(1,    // B (batch)
                                                              32,   // H (heads)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ),
                                        explicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              128,             // S_q (query len)
                                                              128,             // S_kv (kv len)
                                                              64,              // E (embedding)
                                                              64,              // Ev (V embedding)
                                                              {32, 128, 128},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        explicit_transpose_4d(1,                // B (batch)
                                                              32,               // H (heads)
                                                              -1,               // S_q (query len)
                                                              -1,               // S_kv (kv len)
                                                              32,               // E (embedding)
                                                              32,               // Ev (V embedding)
                                                              {1, 32, -1, -1},  // mask_shape
                                                              0.125f            // scale
                                                              ),
                                        explicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              10,              // S_q (query len)
                                                              10,              // S_kv (kv len)
                                                              32,              // E (embedding)
                                                              32,              // Ev (V embedding)
                                                              {1, 1, 10, 10},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        explicit_transpose_4d(1,                 // B (batch)
                                                              10,                // H (heads)
                                                              1024,              // S_q (query len)
                                                              1024,              // S_kv (kv len)
                                                              64,                // E (embedding)
                                                              64,                // Ev (V embedding)
                                                              {10, 1024, 1024},  // mask_shape
                                                              0.125f             // scale
                                                              ))));

class SDPAFusionExplicitTranspose
    : public TransformationTestsF,
      public ::testing::WithParamInterface<std::tuple<Type, bool, bool, SDPAFusionParams>> {};

// Test for SDPAFusion with explicit transpose for K node in the pattern
TEST_P(SDPAFusionExplicitTranspose, SDPAFusionTest_explicit_transpose) {
    // Parametrization
    const auto& type = std::get<0>(GetParam());
    const auto& with_mask = std::get<1>(GetParam());
    const auto& with_scale = std::get<2>(GetParam());
    const auto& param = std::get<3>(GetParam());

    // Init.
    SDPA sdpa(type, param.q_shape, param.k_shape, param.v_shape);
    SDPA sdpa_ref(type, param.q_shape, param.k_shape, param.v_shape);

    // Attention mask processing.
    if (with_mask) {
        sdpa.set_mask(param.mask_shape);
        sdpa_ref.set_mask(param.mask_shape);
    }

    // Scale processing.
    if (with_scale) {
        sdpa.set_scale(param.scale);
        sdpa_ref.set_scale(param.scale);
    }

    // SDPA model.
    {
        sdpa.transpose_k(get_tranpose_order(param.q_shape.size()));
        sdpa.create_pattern_sdpa();
        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        // insert additional transpose, the transposes will be eliminated in another transformation, e.g.
        // TransposeSinking
        sdpa_ref.transpose_k(get_tranpose_order(param.q_shape.size()));
        sdpa_ref.transpose_k(get_tranpose_order(param.q_shape.size()));

        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(SDPAFusion,
                         SDPAFusionExplicitTranspose,
                         Combine(Values(f32, f16),                  // Types
                                 Values(true, false),               // Use attention_mask
                                 Values(true, false),               // Use scale
                                 Values(explicit_transpose_4d(1,    // B (batch)
                                                              32,   // H (heads)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ),
                                        explicit_transpose_4d(1,               // B (batch)
                                                              32,              // H (heads)
                                                              128,             // S_q (query len)
                                                              128,             // S_kv (kv len)
                                                              64,              // E (embedding)
                                                              64,              // Ev (V embedding)
                                                              {32, 128, 128},  // mask_shape
                                                              0.125f           // scale
                                                              ),
                                        explicit_transpose_3d(1,    // B (batch)
                                                              5,    // S_q (query len)
                                                              3,    // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ),
                                        explicit_transpose_4d(-1,   // B (batch)
                                                              32,   // H (heads)
                                                              -1,   // S_q (query len)
                                                              -1,   // S_kv (kv len)
                                                              32,   // E (embedding)
                                                              32,   // Ev (V embedding)
                                                              {},   // mask_shape
                                                              1.0f  // scale
                                                              ))));

TEST_F(TransformationTestsF, SDPAFusionTest7) {
    using namespace ov;
    using namespace ov::element;

    // Init.
    const PartialShape query_shape{1, 8, -1, 32};
    const PartialShape key_shape{-1, 1, 8, 32};
    const PartialShape value_shape{1, 8, -1, 32};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);
    vector<size_t> order = {1, 2, 3, 0};

    // SDPA model.
    {
        sdpa.transpose_k(order);
        sdpa.create_pattern_sdpa();
        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        // Insert additional transpose, the transposes will be eliminated in another transformation, e.g.
        // TransposeSinking
        sdpa_ref.transpose_k(order);
        sdpa_ref.transpose_k(get_tranpose_order(query_shape.size()));

        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest9) {
    // Init.
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 1024, 64};
    const PartialShape value_shape{1, 10, 1024, 64};
    const PartialShape mask_shape{10, 1024, 1024};

    const Shape query_reshaped{10, 1024, 64};
    const Shape key_reshaped{10, 1024, 64};
    const Shape value_reshaped{10, 1024, 64};
    const Shape final_output{1, 10, 1024, 64};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // SDPA model.
    {
        sdpa.set_mask(mask_shape);
        sdpa.reshape_q(query_reshaped);
        sdpa.reshape_k(key_reshaped);
        sdpa.reshape_v(value_reshaped);
        sdpa.transpose_k({0, 2, 1});
        sdpa.set_scale(1.0f);
        sdpa.create_pattern_sdpa();
        sdpa.reshape_sdpa(final_output);

        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_mask(mask_shape);
        sdpa_ref.set_scale(1.0f);
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest10) {
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 77, 64};
    const PartialShape value_shape{1, 10, 77, 64};

    const Shape query_reshaped{10, 1024, 64};
    const Shape key_reshaped{10, 77, 64};
    const Shape value_reshaped{10, 77, 64};
    const Shape final_output{1, 10, 1024, 64};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // SDPA model.
    {
        sdpa.reshape_q(query_reshaped);
        sdpa.reshape_k(key_reshaped);
        sdpa.reshape_v(value_reshaped);
        sdpa.set_scale(1.0f);
        sdpa.create_pattern_sdpa(/*transpose_b=*/true);
        sdpa.reshape_sdpa(final_output);

        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_scale(1.0f);
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest11) {
    // Init.
    const PartialShape query_shape{1, 10, 1024, 64};
    const PartialShape key_shape{1, 10, 77, 64};
    const PartialShape value_shape{1, 10, 77, 64};
    const PartialShape mask_shape{10, 1024, 77};

    const Shape query_reshaped{10, 1024, 64};
    const Shape key_reshaped{10, 77, 64};
    const Shape value_reshaped{10, 77, 64};
    const Shape final_output{1, 10, 1024, 64};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // SDPA model.
    {
        sdpa.set_mask(mask_shape);
        sdpa.reshape_q(query_reshaped);
        sdpa.reshape_k(key_reshaped);
        sdpa.reshape_v(value_reshaped);
        sdpa.transpose_k({0, 2, 1});
        sdpa.set_scale(1.0f);
        sdpa.create_pattern_sdpa();
        sdpa.reshape_sdpa(final_output);

        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_mask(mask_shape);
        sdpa_ref.set_scale(1.0f);
        sdpa_ref.create_reference_sdpa();
        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest12) {
    // Init.
    const PartialShape query_shape{1, 1, 49, 128};
    const PartialShape key_shape{1, 128, 1, 49};
    const PartialShape value_shape{1, 1, 49, 128};

    const PartialShape mask_shape{1, 1, 49, 49};

    const Shape query_reshaped{49, 128};
    const Shape key_reshaped{128, 49};
    const Shape value_reshaped{49, 128};

    const Shape final_output{1, 1, 49, 128};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // SDPA model.
    {
        sdpa.set_mask(mask_shape);
        sdpa.set_scale(1.0f);

        sdpa.reshape_q(query_reshaped);
        sdpa.reshape_k(key_reshaped);
        sdpa.reshape_v(value_reshaped);

        sdpa.create_pattern_sdpa();
        sdpa.reshape_sdpa(final_output);

        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_mask(mask_shape);
        sdpa_ref.set_scale(1.0f);

        sdpa_ref.transpose_k({0, 2, 3, 1});
        sdpa_ref.create_reference_sdpa();

        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest_Split) {
    // Init.
    const PartialShape query_shape{1, 1, 49, 128};
    const PartialShape key_shape{1, 1, 128, 49};
    const PartialShape value_shape{1, 1, 49, 128};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // Preprocessing callback.
    auto callback = [](auto& nodes) {
        auto concat = std::make_shared<v0::Concat>(OutputVector{nodes[InputType::Q], nodes[InputType::V]}, 3);
        auto axis = v0::Constant::create(i64, {}, {3});
        auto split = std::make_shared<v1::Split>(concat, axis, 2);
        nodes[InputType::Q] = split->output(0);
        nodes[InputType::V] = split->output(1);
    };

    // SDPA model.
    {
        sdpa.set_preprocessing_callback(callback);

        sdpa.create_pattern_sdpa();

        model = sdpa.build_model();
        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_preprocessing_callback(callback);

        sdpa_ref.transpose_k({0, 1, 3, 2});
        sdpa_ref.create_reference_sdpa();

        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest_ReshapeOptimization) {
    // Init.
    const PartialShape query_shape{1, 49, 52};
    const PartialShape key_shape{1, 49, 52};
    const PartialShape value_shape{1, 49, 52};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // SDPA model.
    {
        sdpa.reshape_q({1, 49, 1, 52});
        sdpa.reshape_k({1, 49, 1, 52});
        sdpa.reshape_v({1, 49, 1, 52});

        sdpa.reshape_q({1, 1, 49, 52});
        sdpa.reshape_k({1, 1, 49, 52});
        sdpa.reshape_v({1, 1, 49, 52});

        sdpa.reshape_q({1, 49, 52});
        sdpa.reshape_k({1, 49, 52});
        sdpa.reshape_v({1, 49, 52});

        sdpa.create_pattern_sdpa(true);

        sdpa.reshape_sdpa({1, 1, 49, 52});

        model = sdpa.build_model();

        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.reshape_q({1, 49, 1, 52});
        sdpa_ref.reshape_k({1, 49, 1, 52});
        sdpa_ref.reshape_v({1, 49, 1, 52});

        sdpa_ref.reshape_q({1, 1, 49, 52});
        sdpa_ref.reshape_k({1, 1, 49, 52});
        sdpa_ref.reshape_v({1, 1, 49, 52});

        sdpa_ref.create_reference_sdpa();

        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, SDPAFusionTest_4dAttentionMaskWithBatch2) {
    // Init.
    int64_t batch = 2;
    const PartialShape query_shape{batch, 49, 52};
    const PartialShape key_shape{batch, 49, 52};
    const PartialShape value_shape{batch, 49, 52};

    SDPA sdpa(f16, query_shape, key_shape, value_shape);
    SDPA sdpa_ref(f16, query_shape, key_shape, value_shape);

    // Preprocessing callback.
    auto callback = [](auto& nodes) {
        int64_t axis = 0;
        auto axes_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::Q] = std::make_shared<v0::Unsqueeze>(nodes[InputType::Q], axes_node)->output(0);

        auto axes_node1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::K] = std::make_shared<v0::Unsqueeze>(nodes[InputType::K], axes_node1)->output(0);

        auto axes_node2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        nodes[InputType::V] = std::make_shared<v0::Unsqueeze>(nodes[InputType::V], axes_node2)->output(0);
    };

    // SDPA model.
    {
        sdpa.set_mask({batch, 1, 1, 49});
        sdpa.create_pattern_sdpa(true);

        model = sdpa.build_model();

        manager.register_pass<ov::pass::SDPAFusion>();
    }

    // SDPA reference model.
    {
        sdpa_ref.set_mask({batch, 1, 1, 49});
        sdpa_ref.set_preprocessing_callback(callback);
        sdpa_ref.create_reference_sdpa();

        model_ref = sdpa_ref.build_model();
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}