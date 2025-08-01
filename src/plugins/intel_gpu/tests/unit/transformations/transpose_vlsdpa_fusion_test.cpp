// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include <openvino/runtime/core.hpp>
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"

#include "plugin/transformations/transpose_fusion.hpp"

#include "ov_ops/vl_sdpa.hpp"

#include <openvino/pass/serialize.hpp>

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;
using namespace ov;
using namespace ov::opset13;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {
const std::string mask_name = "cu_seq_lens";

std::shared_ptr<ov::Model> build_model() {
    auto q = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});  /* L,H,S */
    auto k = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});
    auto v = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});
    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");

    auto transpose_q = std::make_shared<Transpose>(q, Constant::create(element::i64, Shape{3}, {1,0,2}));
    auto transpose_k = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{3}, {1,0,2}));
    auto transpose_v = std::make_shared<Transpose>(v, Constant::create(element::i64, Shape{3}, {1,0,2}));
    transpose_q->set_friendly_name("transpose_q");
    transpose_k->set_friendly_name("transpose_k");
    transpose_v->set_friendly_name("transpose_v");

    auto cuseq_mask = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    cuseq_mask->set_friendly_name(mask_name);
    cuseq_mask->get_output_tensor(0).set_names({mask_name});

    auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(OutputVector{transpose_q, 
                                                                          transpose_k, 
                                                                          transpose_v,
                                                                          cuseq_mask});

    auto transpose_o = std::make_shared<Transpose>(vlsdpa, Constant::create(element::i64, Shape{3}, {1,0,2}));
    transpose_o->set_friendly_name("transpose_o");

    return std::make_shared<ov::Model>(NodeVector{transpose_o}, ParameterVector{q, k, v, cuseq_mask});
}

std::shared_ptr<ov::Model> build_target_model() {
    auto q = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});  /* L,H,S */
    auto k = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});
    auto v = std::make_shared<Parameter>(element::f32, PartialShape{-1,8,32});

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");

    auto cuseq_mask = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    cuseq_mask->set_friendly_name(mask_name);
    cuseq_mask->get_output_tensor(0).set_names({mask_name});

    auto vlsdpa = std::make_shared<ov::op::internal::VLSDPA>(OutputVector{q, 
                                                                          k, 
                                                                          v,
                                                                          cuseq_mask});

    return std::make_shared<ov::Model>(NodeVector{vlsdpa}, ParameterVector{q, k, v, cuseq_mask});
}
};   // namespace

TEST_F(TransformationTestsF, TransposeVLSDPATest) {
    disable_rt_info_check();
    {
        model = build_model();
        manager.register_pass<TransposeFusion>();
    }
    { model_ref = build_target_model(); }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
