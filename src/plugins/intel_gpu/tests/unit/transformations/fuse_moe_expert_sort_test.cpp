// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_expert_sort.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_expert_sort.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

constexpr size_t hidden_size = 16;
constexpr size_t top_k = 2;
constexpr size_t num_experts = 8;

struct ModelConfig {
    ov::element::Type ids_type = ov::element::i64;
    bool slice_on_updates = true;              // scatter_add_ updates are Slice(Broadcast(1)) as exported by the PT FE
    bool extra_perm_consumer = false;          // the permutation is also a model output
    ov::op::v11::TopK::Mode argsort_mode = ov::op::v11::TopK::Mode::MIN;
};

std::shared_ptr<ov::Node> argsort(const ov::Output<ov::Node>& data, ov::op::v11::TopK::Mode mode = ov::op::v11::TopK::Mode::MIN) {
    auto len = std::make_shared<ov::op::v0::Squeeze>(std::make_shared<ov::op::v3::ShapeOf>(data));
    return std::make_shared<ov::op::v11::TopK>(data, len, -1, mode, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i64);
}

ov::Output<ov::Node> gather0(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& indices) {
    return std::make_shared<ov::op::v8::Gather>(data, indices, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
}

ov::Output<ov::Node> to_i32(const ov::Output<ov::Node>& x) {
    return std::make_shared<ov::op::v0::Convert>(x, ov::element::i32);
}

// Outputs: hidden rows sorted by expert, GroupedMatMul offsets, sorted rows gathered back to the (token, slot) order
std::shared_ptr<ov::Model> build_model(const ModelConfig& cfg) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(cfg.ids_type, ov::PartialShape{-1, top_k});

    auto flat_ids = std::make_shared<ov::op::v1::Reshape>(topk_ids, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto sort = argsort(flat_ids, cfg.argsort_mode);
    auto perm = to_i32(sort->output(1));

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_ids);
    auto num_tokens = std::make_shared<ov::op::v8::Gather>(ids_shape,
                                                           ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                           ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto range = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                     num_tokens,
                                                     ov::op::v0::Constant::create(ov::element::i64, {}, {1}),
                                                     ov::element::i64);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(range, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
    auto expand = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, ids_shape);
    auto token_src = std::make_shared<ov::op::v1::Reshape>(expand, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto sorted_hidden = gather0(hidden, to_i32(gather0(token_src, perm)));

    auto sorted_ids = to_i32(gather0(flat_ids, perm));
    ov::Output<ov::Node> updates =
        std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(ov::element::i32, {}, {1}), std::make_shared<ov::op::v3::ShapeOf>(perm));
    if (cfg.slice_on_updates) {
        updates = std::make_shared<ov::op::v8::Slice>(updates,
                                                      ov::op::v0::Constant::create(ov::element::i32, {1}, {0}),
                                                      std::make_shared<ov::op::v3::ShapeOf>(sorted_ids, ov::element::i32),
                                                      ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                                                      ov::op::v0::Constant::create(ov::element::i32, {1}, {0}));
    }
    auto counts = std::make_shared<ov::op::v12::ScatterElementsUpdate>(ov::op::v0::Constant::create(ov::element::i32, {num_experts}, {0}),
                                                                       sorted_ids,
                                                                       updates,
                                                                       ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                                       ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
    auto offsets = std::make_shared<ov::op::v0::CumSum>(counts, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    offsets->set_friendly_name("offsets");

    auto inv_perm = argsort(sort->output(1))->output(1);
    auto unsorted_hidden = gather0(sorted_hidden, to_i32(inv_perm));

    ov::OutputVector outputs{sorted_hidden, offsets, unsorted_hidden};
    if (cfg.extra_perm_consumer) {
        outputs.push_back(perm);
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{hidden, topk_ids});
}

std::shared_ptr<ov::Model> build_model_ref(const ModelConfig& cfg) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(cfg.ids_type, ov::PartialShape{-1, top_k});

    auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(topk_ids, num_experts);
    sort->set_friendly_name("offsets");
    auto token = std::make_shared<ov::op::v0::Convert>(sort->output(0), ov::element::i64);
    auto sorted_hidden = gather0(hidden, to_i32(token));
    auto inv_perm = std::make_shared<ov::op::v0::Convert>(sort->output(2), ov::element::i64);
    auto unsorted_hidden = gather0(sorted_hidden, to_i32(inv_perm));
    return std::make_shared<ov::Model>(ov::OutputVector{sorted_hidden, sort->output(1), unsorted_hidden}, ov::ParameterVector{hidden, topk_ids});
}

}  // namespace

using FuseMoEExpertSortParams = std::tuple<ov::element::Type, bool>;  // ids type, Slice on the scatter updates

class FuseMoEExpertSortTest : public TransformationTestsF, public WithParamInterface<FuseMoEExpertSortParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<FuseMoEExpertSortParams>& info) {
        const auto& [ids_type, slice_on_updates] = info.param;
        return "ids_" + ids_type.get_type_name() + (slice_on_updates ? "_SliceUpdates" : "_BroadcastUpdates");
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [ids_type, slice_on_updates] = GetParam();
        ModelConfig cfg;
        cfg.ids_type = ids_type;
        cfg.slice_on_updates = slice_on_updates;
        model = build_model(cfg);
        model_ref = build_model_ref(cfg);
        manager.register_pass<FuseMoEExpertSort>(num_experts);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(FuseMoEExpertSortTest, Fused) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         FuseMoEExpertSortTest,
                         Combine(Values(ov::element::i64, ov::element::i32), Values(true, false)),
                         FuseMoEExpertSortTest::get_test_case_name);

TEST_F(TransformationTestsF, FuseMoEExpertSort_PermutationHasOtherConsumer) {
    ModelConfig cfg;
    cfg.extra_perm_consumer = true;
    model = build_model(cfg);
    manager.register_pass<FuseMoEExpertSort>(num_experts);
}

TEST_F(TransformationTestsF, FuseMoEExpertSort_TooManyExperts) {
    model = build_model(ModelConfig{});
    manager.register_pass<FuseMoEExpertSort>(num_experts - 1);
}

TEST_F(TransformationTestsF, FuseMoEExpertSort_DescendingArgsort) {
    ModelConfig cfg;
    cfg.argsort_mode = ov::op::v11::TopK::Mode::MAX;
    model = build_model(cfg);
    manager.register_pass<FuseMoEExpertSort>(num_experts);
}

}  // namespace ov::test::intel_gpu
