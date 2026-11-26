// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_expert_sort.hpp"

#include <gtest/gtest.h>

#include <sstream>

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
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

constexpr size_t hidden_size = 16;
constexpr size_t top_k = 2;
constexpr size_t num_experts = 8;

// ids type,
// scatter_add_ updates are Slice(Broadcast(1)) as exported by the PT FE,
// the hidden rows are gathered by the token index of the permutation,
// rows are gathered back by the argsort of the permutation,
// the permutation is also a model output,
// argsort mode,
// max number of experts of the pass,
// top_k the token source is expanded to,
// the permutation is converted to i32,
// the number of tokens of the token source is taken from the hidden states instead of the ids
using FuseMoEExpertSortParams = std::tuple<ov::element::Type, bool, bool, bool, bool, ov::op::v11::TopK::Mode, size_t, size_t, bool, bool>;

std::shared_ptr<ov::Node> argsort(const ov::Output<ov::Node>& data, ov::op::v11::TopK::Mode mode = ov::op::v11::TopK::Mode::MIN) {
    auto len = std::make_shared<ov::op::v0::Squeeze>(std::make_shared<ov::op::v3::ShapeOf>(data));
    return std::make_shared<ov::op::v11::TopK>(data, len, -1, mode, ov::op::v11::TopK::SortType::SORT_VALUES, ov::element::i64);
}

// Outputs: hidden rows sorted by expert, GroupedMatMul offsets, sorted rows gathered back to the (token, slot) order
std::shared_ptr<ov::Model> build_model(ov::element::Type ids_type,
                                       bool slice_on_updates,
                                       bool token_path,
                                       bool inverse_perm,
                                       bool extra_perm_consumer,
                                       ov::op::v11::TopK::Mode argsort_mode,
                                       size_t expand_top_k,
                                       bool perm_to_i32,
                                       bool tokens_from_hidden) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ids_type, ov::PartialShape{-1, top_k});
    auto axis0 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});

    auto flat_ids = std::make_shared<ov::op::v1::Reshape>(topk_ids, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto sort = argsort(flat_ids, argsort_mode);
    ov::Output<ov::Node> perm = sort->output(1);
    if (perm_to_i32) {
        perm = std::make_shared<ov::op::v0::Convert>(perm, ov::element::i32);
    }

    ov::Output<ov::Node> sorted_hidden = hidden;
    if (token_path) {
        ov::Output<ov::Node> tokens_source = tokens_from_hidden ? ov::Output<ov::Node>(hidden) : ov::Output<ov::Node>(topk_ids);
        auto num_tokens = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(tokens_source),
                                                               ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                               ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto range = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant::create(ov::element::i64, {}, {0}),
                                                         num_tokens,
                                                         ov::op::v0::Constant::create(ov::element::i64, {}, {1}),
                                                         ov::element::i64);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(range, ov::op::v0::Constant::create(ov::element::i64, {}, {1}));
        auto expand = std::make_shared<ov::op::v3::Broadcast>(unsqueeze,
                                                              ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<size_t>{1, expand_top_k}),
                                                              ov::op::BroadcastType::BIDIRECTIONAL);
        auto token_src = std::make_shared<ov::op::v1::Reshape>(expand, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
        auto token = std::make_shared<ov::op::v8::Gather>(token_src, perm, axis0);
        sorted_hidden = std::make_shared<ov::op::v8::Gather>(hidden, std::make_shared<ov::op::v0::Convert>(token, ov::element::i32), axis0);
    }

    auto sorted_ids = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v8::Gather>(flat_ids, perm, axis0), ov::element::i32);
    ov::Output<ov::Node> updates =
        std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(ov::element::i32, {}, {1}), std::make_shared<ov::op::v3::ShapeOf>(perm));
    if (slice_on_updates) {
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

    ov::OutputVector outputs{sorted_hidden, offsets};
    if (inverse_perm) {
        auto inv_perm = std::make_shared<ov::op::v0::Convert>(argsort(sort->output(1))->output(1), ov::element::i32);
        outputs.push_back(std::make_shared<ov::op::v8::Gather>(sorted_hidden, inv_perm, axis0));
    }
    if (extra_perm_consumer) {
        outputs.push_back(perm);
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{hidden, topk_ids});
}

std::shared_ptr<ov::Model> build_model_ref(ov::element::Type ids_type, bool token_path, bool inverse_perm) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ids_type, ov::PartialShape{-1, top_k});
    auto axis0 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});

    auto sort = std::make_shared<ov::intel_gpu::op::MoEExpertSort>(topk_ids, num_experts);
    sort->set_friendly_name("offsets");
    ov::Output<ov::Node> sorted_hidden = hidden;
    if (token_path) {
        auto token = std::make_shared<ov::op::v0::Convert>(sort->output(0), ov::element::i64);
        sorted_hidden = std::make_shared<ov::op::v8::Gather>(hidden, std::make_shared<ov::op::v0::Convert>(token, ov::element::i32), axis0);
    }
    ov::OutputVector outputs{sorted_hidden, sort->output(1)};
    if (inverse_perm) {
        auto inv_perm = std::make_shared<ov::op::v0::Convert>(sort->output(2), ov::element::i64);
        outputs.push_back(std::make_shared<ov::op::v8::Gather>(sorted_hidden, std::make_shared<ov::op::v0::Convert>(inv_perm, ov::element::i32), axis0));
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{hidden, topk_ids});
}

}  // namespace

class FuseMoEExpertSortTest : public TransformationTestsF, public WithParamInterface<FuseMoEExpertSortParams> {
public:
    static std::string get_test_case_name(const TestParamInfo<FuseMoEExpertSortParams>& info) {
        const auto& [ids_type,
                     slice_on_updates,
                     token_path,
                     inverse_perm,
                     extra_perm_consumer,
                     argsort_mode,
                     max_experts,
                     expand_top_k,
                     perm_to_i32,
                     tokens_from_hidden] = info.param;
        std::ostringstream name;
        name << "ids_" << ids_type.get_type_name() << (slice_on_updates ? "_SliceUpdates" : "_BroadcastUpdates") << (token_path ? "_Token" : "_NoToken")
             << (inverse_perm ? "_Inverse" : "_NoInverse");
        if (extra_perm_consumer) {
            name << "_PermutationHasOtherConsumer";
        }
        if (argsort_mode == ov::op::v11::TopK::Mode::MAX) {
            name << "_DescendingArgsort";
        }
        if (max_experts < num_experts) {
            name << "_TooManyExperts";
        }
        if (expand_top_k != top_k) {
            name << "_TokenSourceTopKMismatch";
        }
        if (tokens_from_hidden) {
            name << "_TokenSourceNumTokensMismatch";
        }
        name << (perm_to_i32 ? "_PermI32" : "_PermI64");
        return name.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [ids_type,
                     slice_on_updates,
                     token_path,
                     inverse_perm,
                     extra_perm_consumer,
                     argsort_mode,
                     max_experts,
                     expand_top_k,
                     perm_to_i32,
                     tokens_from_hidden] = GetParam();
        model =
            build_model(ids_type, slice_on_updates, token_path, inverse_perm, extra_perm_consumer, argsort_mode, expand_top_k, perm_to_i32, tokens_from_hidden);
        const bool fused =
            !extra_perm_consumer && argsort_mode == ov::op::v11::TopK::Mode::MIN && max_experts >= num_experts && expand_top_k == top_k && !tokens_from_hidden;
        if (fused) {
            model_ref = build_model_ref(ids_type, token_path, inverse_perm);
        }
        // Validation after a pass would drop the symbols
        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::SymbolicPropagation>();
        manager.register_pass<FuseMoEExpertSort>(max_experts);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(FuseMoEExpertSortTest, Transform) {}

// The token index and the inverse permutation are optional consumers of the sort
INSTANTIATE_TEST_SUITE_P(smoke_Fused,
                         FuseMoEExpertSortTest,
                         Combine(Values(ov::element::i64, ov::element::i32),
                                 Bool(),
                                 Bool(),
                                 Bool(),
                                 Values(false),
                                 Values(ov::op::v11::TopK::Mode::MIN),
                                 Values(num_experts),
                                 Values(top_k),
                                 Bool(),
                                 Values(false)),
                         FuseMoEExpertSortTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(
    smoke_NotFused,
    FuseMoEExpertSortTest,
    Values(
        // the permutation is also a model output
        FuseMoEExpertSortParams{ov::element::i64, true, true, true, true, ov::op::v11::TopK::Mode::MIN, num_experts, top_k, true, false},
        // the sort is descending
        FuseMoEExpertSortParams{ov::element::i64, true, true, true, false, ov::op::v11::TopK::Mode::MAX, num_experts, top_k, true, false},
        // the kernel supports fewer experts
        FuseMoEExpertSortParams{ov::element::i64, true, true, true, false, ov::op::v11::TopK::Mode::MIN, num_experts - 1, top_k, true, false},
        // the token source is expanded to another top_k than the ids have
        FuseMoEExpertSortParams{ov::element::i64, true, true, true, false, ov::op::v11::TopK::Mode::MIN, num_experts, top_k + 1, true, false},
        // the token source has the number of tokens of another tensor than the ids
        FuseMoEExpertSortParams{ov::element::i64, true, true, true, false, ov::op::v11::TopK::Mode::MIN, num_experts, top_k, true, true}),
    FuseMoEExpertSortTest::get_test_case_name);

}  // namespace ov::test::intel_gpu
