// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/sdpa_transpose_fusion.hpp"

#include <memory>
#include <vector>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::shared_ptr<ov::op::v1::Transpose> make_transpose(const ov::Output<ov::Node>& input,
                                                              const std::vector<int64_t>& order) {
    auto order_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{order.size()}, order);
    return std::make_shared<ov::op::v1::Transpose>(input, order_c);
}

static std::shared_ptr<ov::intel_gpu::op::SDPA> make_internal_sdpa(
    const ov::Output<ov::Node>& q,
    const ov::Output<ov::Node>& k,
    const ov::Output<ov::Node>& v,
    bool causal,
    const std::vector<int64_t>& in0_order,
    const std::vector<int64_t>& in1_order,
    const std::vector<int64_t>& in2_order,
    const std::vector<int64_t>& out_order) {
    return std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{q, k, v},
                                                     causal,
                                                     in0_order,
                                                     in1_order,
                                                     in2_order,
                                                     out_order,
                                                     ov::element::dynamic);
}

// ---------------------------------------------------------------------------
// 1. Internal op::SDPA (identity output order) + Transpose({0,2,1,3})
//    → SDPA with output_order = {0,2,1,3}, Transpose eliminated.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_InternalSDPA_OutputTransposeFused) {
    const bool causal = false;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, identity);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // Transpose absorbed: output order becomes {0,2,1,3}
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 2. Internal op::SDPA (identity output order) + Transpose({0,2,1,3})
//    causal=true variant — same fusion should apply regardless of causal flag.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_InternalSDPA_Causal_OutputTransposeFused) {
    const bool causal = true;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, identity);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 3. Internal op::SDPA with non-identity output order already set
//    → must NOT fuse (pass guards against double-applying).
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_InternalSDPA_NonIdentityOutputOrder_NoFusion) {
    const bool causal = false;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> custom_out{0, 2, 1, 3};  // already non-identity
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // SDPA already has a non-identity output order
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, custom_out);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        // Graph must be unchanged — reference equals input
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, custom_out);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 4. Internal op::SDPA followed by a non-{0,2,1,3} Transpose
//    → must NOT fuse (only the heads<->seq swap is absorbed).
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_InternalSDPA_WrongTransposeOrder_NoFusion) {
    const bool causal = false;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> wrong_order{0, 3, 1, 2};  // not the heads<->seq swap

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, identity);
        auto out_tp = make_transpose(sdpa, wrong_order);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, identity);
        auto out_tp = make_transpose(sdpa, wrong_order);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 5. Framework v13::ScaledDotProductAttention + Transpose({0,2,1,3})
//    → converted to internal op::SDPA with identity input orders and
//      output_order = {0,2,1,3}.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_V13SDPA_OutputTransposeFused) {
    const bool causal = false;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // v13 SDPA promoted to internal SDPA with identity input orders;
        // output Transpose absorbed into output_order.
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 6. Framework v13::SDPA + causal + Transpose({0,2,1,3})
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_V13SDPA_Causal_OutputTransposeFused) {
    const bool causal = true;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, identity, identity, identity, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 7. Framework v13::SDPA with input Transposes (pi05 pattern) +
//    Transpose({0,2,1,3}) on output.
//    Input transposes must be left standalone; only the output is absorbed.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_V13SDPA_WithInputTransposes_OnlyOutputFused) {
    const bool causal = false;
    const std::vector<int64_t> identity{0, 1, 2, 3};
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};
    const std::vector<int64_t> k_perm{0, 1, 3, 2};  // head_size-moving — must stay explicit

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto q_tp = make_transpose(q, swap_hs);
        auto k_tp = make_transpose(k, k_perm);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_tp, k_tp, v, causal);
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // Input transposes stay explicit; output Transpose absorbed into SDPA.
        auto q_tp = make_transpose(q, swap_hs);
        auto k_tp = make_transpose(k, k_perm);
        auto sdpa = make_internal_sdpa(q_tp, k_tp, v, causal, identity, identity, identity, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 8. Framework v13::SDPA with no output Transpose
//    → no fusion, graph unchanged.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_V13SDPA_NoOutputTranspose_NoFusion) {
    const bool causal = false;

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, causal);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// ---------------------------------------------------------------------------
// 9. Internal op::SDPA with non-identity input orders + Transpose({0,2,1,3})
//    Input orders must be preserved after fusion.
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, SDPATransposeFusion_InternalSDPA_NonIdentityInputOrders_Preserved) {
    const bool causal = false;
    const std::vector<int64_t> swap_hs{0, 2, 1, 3};
    const std::vector<int64_t> q_order{0, 2, 1, 3};
    const std::vector<int64_t> k_order{0, 2, 3, 1};
    const std::vector<int64_t> v_order{0, 1, 2, 3};

    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto sdpa = make_internal_sdpa(q, k, v, causal, q_order, k_order, v_order, {0, 1, 2, 3});
        auto out_tp = make_transpose(sdpa, swap_hs);

        model = std::make_shared<ov::Model>(ov::OutputVector{out_tp}, ov::ParameterVector{q, k, v});
        manager.register_pass<SDPATransposeFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        // Input orders preserved; output order absorbs the Transpose.
        auto sdpa = make_internal_sdpa(q, k, v, causal, q_order, k_order, v_order, swap_hs);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
