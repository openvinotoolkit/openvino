// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/indirect_kv_cache.hpp"
#include "plugin/transformations/kv_cache_fusion.hpp"
#include "plugin/transformations/transpose_fusion.hpp"
#include "plugin/transformations/unsqueeze_broadcast_reshape_sdpa_fusion.hpp"
#include "subgraphs_builders.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {
namespace {

std::shared_ptr<ov::Model> make_ref_model(ov::Dimension batch = ov::Dimension::dynamic(),
                                          ov::Dimension n_heads = ov::Dimension::dynamic(),
                                          ov::Dimension n_features = ov::Dimension::dynamic(),
                                          ov::element::Type_t element_type = ov::element::f32,
                                          std::vector<int64_t> qkv_order = {0, 1, 2, 3},
                                          bool causal = false,
                                          bool with_mask = false,
                                          bool with_scale = false,
                                          bool stateful = false,
                                          bool with_rearrange = false,
                                          size_t num_groups = 1) {
    ov::PartialShape kv_cache_size_def = {batch, n_heads / num_groups, -1, n_features};
    ov::PartialShape new_token_size_def = {batch, n_heads / num_groups, -1, n_features};
    ov::PartialShape q_size_def = {batch, n_heads, -1, n_features};

    ov::PartialShape kv_cache_size = ov::PartialShape::dynamic(4);
    ov::PartialShape new_token_size = ov::PartialShape::dynamic(4);
    ov::PartialShape q_size = ov::PartialShape::dynamic(4);

    for (size_t i = 0; i < kv_cache_size_def.size(); i++) {
        kv_cache_size[qkv_order[i]] = kv_cache_size_def[i];
        new_token_size[qkv_order[i]] = new_token_size_def[i];
        q_size[qkv_order[i]] = q_size_def[i];
    }

    int64_t concat_axis = qkv_order[2];

    auto in_q = std::make_shared<ov::op::v0::Parameter>(element_type, q_size);
    auto in_k_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);
    auto in_v_token = std::make_shared<ov::op::v0::Parameter>(element_type, new_token_size);

    ov::op::util::VariableInfo vi1 = {kv_cache_size, element_type, "v1"};
    ov::op::util::VariableInfo vi2 = {kv_cache_size, element_type, "v2"};
    auto v1 = std::make_shared<ov::op::util::Variable>(vi1);
    auto v2 = std::make_shared<ov::op::util::Variable>(vi2);
    auto state_initializer = ::tests::make_state_initializer(in_v_token, element_type, kv_cache_size, qkv_order);
    auto past_k = std::make_shared<ov::intel_gpu::op::ReadValue>(state_initializer, v1);
    auto past_v = std::make_shared<ov::intel_gpu::op::ReadValue>(state_initializer, v2);

    ov::ParameterVector params{in_q, in_k_token, in_v_token};

    std::shared_ptr<ov::Node> q = in_q;
    std::shared_ptr<ov::intel_gpu::op::KVCache> k = nullptr;
    std::shared_ptr<ov::intel_gpu::op::KVCache> v = nullptr;
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;

    if (with_rearrange) {
        auto in_beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
        params.push_back(in_beam_idx);
        k = std::make_shared<ov::intel_gpu::op::KVCache>(past_k, in_k_token, in_beam_idx, v1, concat_axis, 0);
        v = std::make_shared<ov::intel_gpu::op::KVCache>(past_v, in_v_token, in_beam_idx, v2, concat_axis, 0);
    } else {
        k = std::make_shared<ov::intel_gpu::op::KVCache>(past_k, in_k_token, v1, concat_axis);
        v = std::make_shared<ov::intel_gpu::op::KVCache>(past_v, in_v_token, v2, concat_axis);
    }

    if (with_mask) {
        mask = ::tests::make_attention_mask(q, k, element_type, qkv_order);
    }

    if (with_mask && with_scale) {
        scale = ov::op::v0::Constant::create(element_type, ov::Shape{}, { 1.0f / std::sqrt(n_features.get_max_length())});
    }

    std::shared_ptr<ov::Node> sdpa = nullptr;
    ov::OutputVector inputs = {q, k, v};
    if (mask) {
        inputs.push_back(mask);
    }
    if (scale) {
        inputs.push_back(scale);
    }

    if (with_rearrange) {
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(inputs,
                                                                 k->output(1),
                                                                 causal,
                                                                 concat_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));
    } else {
        sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                         causal,
                                                         qkv_order,
                                                         qkv_order,
                                                         qkv_order,
                                                         ov::intel_gpu::op::SDPA::default_order(4));
    }

    auto sdpa_out = std::make_shared<ov::op::v0::Result>(sdpa);

    ov::ResultVector results{sdpa_out};
    auto model = std::make_shared<ov::Model>(results, params);

    model->validate_nodes_and_infer_types();

    return model;
}

using Params = std::tuple<bool, bool, bool, bool, ov::Dimension, ov::element::Type, size_t, std::vector<int64_t>>;
class SDPAOptimizationTestsP : public TransformationTestsF, public WithParamInterface<Params> {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
    }

    static std::string get_test_case_name(testing::TestParamInfo<Params> obj) {
        bool with_rearrange;
        bool with_mask;
        bool with_scale;
        bool causal;
        ov::Dimension batch;
        ov::element::Type model_element_type;
        size_t num_groups;
        std::vector<int64_t> qkv_order;
        std::tie(with_rearrange, with_mask, with_scale, causal, batch, model_element_type, num_groups, qkv_order) = obj.param;
        std::ostringstream result;
        result << "with_rearrange=" << with_rearrange << "_";
        result << "with_mask=" << with_mask << "_";
        result << "with_scale=" << with_scale << "_";
        result << "causal=" << causal << "_";
        result << "batch=" << (batch.is_static() ? batch.get_length() : -1) << "_";
        result << "model_element_type=" << model_element_type << "_";
        result << "num_groups=" << num_groups << "_";
        result << "qkv_order=" << ov::test::utils::vec2str(qkv_order);
        return result.str();
    }
};

TEST_P(SDPAOptimizationTestsP, PassesSequence) {
    bool with_rearrange;
    bool with_mask;
    bool with_scale;
    bool causal;
    ov::Dimension batch;
    ov::element::Type model_element_type;
    size_t num_groups;
    std::vector<int64_t> qkv_order;

    std::tie(with_rearrange, with_mask, with_scale, causal, batch, model_element_type, num_groups, qkv_order) = GetParam();
    model = tests::make_llm_kv_cache_sdpa_pattern(batch,
                                                  32,
                                                  128,
                                                  model_element_type,
                                                  qkv_order,
                                                  causal,
                                                  with_mask,
                                                  with_scale,
                                                  true,
                                                  with_rearrange,
                                                  num_groups);

    manager.register_pass<KVCacheFusion>();
    manager.register_pass<TransposeFusion>(false);
    manager.register_pass<UnsqueezeBroadcastReshapeSDPAFusion>();
    manager.register_pass<IndirectKVCache>();

    model_ref = make_ref_model(batch,
                               32,
                               128,
                               model_element_type,
                               qkv_order,
                               causal,
                               with_mask,
                               with_scale,
                               true,
                               with_rearrange,
                               num_groups);
}

const std::vector<bool> with_rearrange_v = {true, false};
const std::vector<bool> with_mask_v = {true, false};
const std::vector<bool> with_scale_v = {true, false};
const std::vector<bool> causal_v = {true, false};
const std::vector<ov::Dimension> batch_v = {-1, 1, 8};
const std::vector<ov::element::Type> dt_v = {ov::element::f16, ov::element::f32};
const std::vector<size_t> num_groups_v = {1, 2};
const std::vector<std::vector<int64_t>> qkv_order_v = {{0, 1, 2, 3}, {0, 2, 1, 3}};

INSTANTIATE_TEST_SUITE_P(smoke,
                         SDPAOptimizationTestsP,
                         testing::Combine(::testing::ValuesIn(with_rearrange_v),
                                          ::testing::ValuesIn(with_mask_v),
                                          ::testing::ValuesIn(with_scale_v),
                                          ::testing::ValuesIn(causal_v),
                                          ::testing::ValuesIn(batch_v),
                                          ::testing::ValuesIn(dt_v),
                                          ::testing::ValuesIn(num_groups_v),
                                          ::testing::ValuesIn(qkv_order_v)),
                         SDPAOptimizationTestsP::get_test_case_name);

}  // namespace
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
