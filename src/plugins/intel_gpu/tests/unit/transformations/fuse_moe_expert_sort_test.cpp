// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_expert_sort.hpp"

#include <gtest/gtest.h>

#include <sstream>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_expert_sort.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

using FuseMoEExpertSortParams = std::tuple<ov::element::Type,  // router ids type
                                           ov::PartialShape,   // router ids shape
                                           size_t,             // number of experts
                                           bool,               // the scatter-add updates are Slice(Broadcast(1)), as exported by the PT FE
                                           bool,               // the hidden rows are gathered by the token index
                                           bool,               // the rows are gathered back by the inverse permutation
                                           bool,               // the permutation is also a model output
                                           bool,               // the argsort is descending
                                           size_t,             // max number of experts of the pass
                                           size_t,             // top_k the token source is expanded to
                                           bool,               // the permutation is converted to i32
                                           bool,               // the token source takes the number of tokens from the hidden states
                                           bool>;              // the pass is expected to fuse

// Outputs: hidden rows sorted by expert, GroupedMatMul offsets, sorted rows gathered back to the (token, slot) order
std::shared_ptr<ov::Model> build_model(const FuseMoEExpertSortParams& params) {
    const auto& [ids_type,
                 ids_shape,
                 num_experts,
                 slice_on_updates,
                 token_path,
                 inverse_perm,
                 perm_output,
                 descending_argsort,
                 max_experts,
                 expand_top_k,
                 perm_to_i32,
                 tokens_from_hidden,
                 fused] = params;
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(2));
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ids_type, ids_shape);
    auto axis0 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});

    ov::test::MoEExpertSortOptions options;
    options.token_index = token_path;
    options.inverse_permutation = inverse_perm;
    options.descending_argsort = descending_argsort;
    options.slice_on_updates = slice_on_updates;
    options.permutation_to_i32 = perm_to_i32;
    options.token_source_top_k = expand_top_k;
    if (tokens_from_hidden) {
        options.token_source_tokens = hidden;
    }
    const auto sort = ov::test::build_moe_expert_sort_subgraph(topk_ids, num_experts, options);
    sort.offsets.get_node()->set_friendly_name("offsets");

    ov::Output<ov::Node> sorted_hidden = hidden;
    if (token_path) {
        sorted_hidden = std::make_shared<ov::op::v8::Gather>(hidden, std::make_shared<ov::op::v0::Convert>(sort.token_index, ov::element::i32), axis0);
    }
    ov::OutputVector outputs{sorted_hidden, sort.offsets};
    if (inverse_perm) {
        auto inv_perm = std::make_shared<ov::op::v0::Convert>(sort.inverse_permutation, ov::element::i32);
        outputs.push_back(std::make_shared<ov::op::v8::Gather>(sorted_hidden, inv_perm, axis0));
    }
    if (perm_output) {
        outputs.push_back(sort.permutation);
    }
    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{hidden, topk_ids});
}

std::shared_ptr<ov::Model> build_model_ref(const FuseMoEExpertSortParams& params) {
    const auto& [ids_type,
                 ids_shape,
                 num_experts,
                 slice_on_updates,
                 token_path,
                 inverse_perm,
                 perm_output,
                 descending_argsort,
                 max_experts,
                 expand_top_k,
                 perm_to_i32,
                 tokens_from_hidden,
                 fused] = params;
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(2));
    auto topk_ids = std::make_shared<ov::op::v0::Parameter>(ids_type, ids_shape);
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
                     ids_shape,
                     num_experts,
                     slice_on_updates,
                     token_path,
                     inverse_perm,
                     perm_output,
                     descending_argsort,
                     max_experts,
                     expand_top_k,
                     perm_to_i32,
                     tokens_from_hidden,
                     fused] = info.param;
        std::ostringstream name;
        name << "ids_" << ids_type.get_type_name() << "_" << ids_shape << "_E" << num_experts << (slice_on_updates ? "_SliceUpdates" : "_BroadcastUpdates")
             << (token_path ? "_Token" : "_NoToken") << (inverse_perm ? "_Inverse" : "_NoInverse");
        if (perm_output) {
            name << "_PermutationHasOtherConsumer";
        }
        if (descending_argsort) {
            name << "_DescendingArgsort";
        }
        name << "_MaxExperts" << max_experts << "_TokenSourceTopK" << expand_top_k;
        if (tokens_from_hidden) {
            name << "_TokenSourceNumTokensMismatch";
        }
        name << (perm_to_i32 ? "_PermI32" : "_PermI64");
        return name.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const bool fused = std::get<12>(GetParam());
        model = build_model(GetParam());
        if (fused) {
            model_ref = build_model_ref(GetParam());
        }
        // Validation after a pass would drop the symbols
        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::SymbolicPropagation>();
        manager.register_pass<FuseMoEExpertSort>(std::get<8>(GetParam()));
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
};

TEST_P(FuseMoEExpertSortTest, Transform) {}

// The token index and the inverse permutation are optional consumers of the sort
INSTANTIATE_TEST_SUITE_P(smoke_Fused,
                         FuseMoEExpertSortTest,
                         Combine(Values(ov::element::i64, ov::element::i32),
                                 Values(ov::PartialShape{-1, 2}),
                                 Values(8),
                                 Bool(),
                                 Bool(),
                                 Bool(),
                                 Values(false),
                                 Values(false),
                                 Values(8),
                                 Values(2),
                                 Bool(),
                                 Values(false),
                                 Values(true)),
                         FuseMoEExpertSortTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_NotFused,
                         FuseMoEExpertSortTest,
                         Values(
                             // the permutation is also a model output
                             FuseMoEExpertSortParams{ov::element::i64, ov::PartialShape{-1, 2}, 8, true, true, true, true, false, 8, 2, true, false, false},
                             // the sort is descending
                             FuseMoEExpertSortParams{ov::element::i64, ov::PartialShape{-1, 2}, 8, true, true, true, false, true, 8, 2, true, false, false},
                             // the kernel supports fewer experts
                             FuseMoEExpertSortParams{ov::element::i64, ov::PartialShape{-1, 2}, 8, true, true, true, false, false, 7, 2, true, false, false},
                             // the token source is expanded to another top_k than the ids have
                             FuseMoEExpertSortParams{ov::element::i64, ov::PartialShape{-1, 2}, 8, true, true, true, false, false, 8, 3, true, false, false},
                             // the token source has the number of tokens of another tensor than the ids
                             FuseMoEExpertSortParams{ov::element::i64, ov::PartialShape{-1, 2}, 8, true, true, true, false, false, 8, 2, true, true, false}),
                         FuseMoEExpertSortTest::get_test_case_name);

}  // namespace ov::test::intel_gpu
