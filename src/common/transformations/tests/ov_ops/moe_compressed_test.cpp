// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/moe_compressed.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>

#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using ov::op::internal::MOE;
using ov::op::internal::MOECompressed;

constexpr size_t kTokens = 2;
constexpr size_t kHiddenSize = 8;
constexpr size_t kInterSize = 8;
constexpr size_t kNumExperts = 2;
constexpr size_t kTopK = 2;

// Builds a minimal valid GEMM3_SWIGLU MOECompressed model (12 inputs, no zero-points) with a config whose
// MOE-base fields (expert_type/expert_alpha/expert_beta/gate_idx/activation_type) are set to non-default
// values, so that a bug losing them on serialize/deserialize or clone would be observable.
std::shared_ptr<ov::Model> build_moe_compressed_model(MOECompressed::Config& config_out) {
    using ov::element::f16;
    using ov::element::u4;
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    using ov::op::v0::Result;

    auto hidden_states = std::make_shared<Parameter>(f16, ov::Shape{kTokens, kHiddenSize});
    auto routing_weights = std::make_shared<Parameter>(f16, ov::Shape{kTokens, kTopK});
    auto topk_idx = std::make_shared<Parameter>(ov::element::i32, ov::Shape{kTokens, kTopK});

    auto make_weight = [](size_t ofm, size_t k) {
        return Constant::create(u4, ov::Shape{kNumExperts, ofm, 1, k}, {1});
    };
    auto make_scale = [](size_t ofm) {
        return Constant::create(f16, ov::Shape{kNumExperts, ofm, 1, 1}, {0.01f});
    };
    // has_zp == false, so zero-point placeholders must carry a dynamic element type and be empty
    // (count == 0), which is how MOECompressed::validate_and_infer_types() recognizes an absent input.
    // They must still be declared as model Parameters, since ov::Model rejects dangling ones.
    ov::ParameterVector zp_placeholders;
    auto make_zp_placeholder = [&zp_placeholders] {
        auto zp = std::make_shared<Parameter>(ov::element::dynamic, ov::Shape{0});
        zp_placeholders.push_back(zp);
        return zp;
    };

    ov::OutputVector args{hidden_states->output(0),
                          routing_weights->output(0),
                          topk_idx->output(0),
                          make_weight(kHiddenSize, kHiddenSize),
                          make_scale(kHiddenSize),
                          make_zp_placeholder()->output(0),
                          make_weight(kHiddenSize, kHiddenSize),
                          make_scale(kHiddenSize),
                          make_zp_placeholder()->output(0),
                          make_weight(kInterSize, kInterSize),
                          make_scale(kInterSize),
                          make_zp_placeholder()->output(0)};

    MOECompressed::Config config;
    config.expert_type = MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = MOE::Activation_type::SWIGLU;
    config.expert_alpha = 0.5f;
    config.expert_beta = 1.5f;
    config.gate_idx = 1;
    config.hidden_size = kHiddenSize;
    config.inter_size = kInterSize;
    config.num_expert = kNumExperts;
    config.top_k = kTopK;
    config.group_size = kHiddenSize;
    config.has_zp = false;
    config.out_type = ov::element::dynamic;

    auto moe = std::make_shared<MOECompressed>(args, config);
    config_out = config;

    ov::ParameterVector params{hidden_states, routing_weights, topk_idx};
    params.insert(params.end(), zp_placeholders.begin(), zp_placeholders.end());
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<Result>(moe->output(0))}, params);
}

std::shared_ptr<MOECompressed> find_moe_compressed(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto moe = std::dynamic_pointer_cast<MOECompressed>(node)) {
            return moe;
        }
    }
    return nullptr;
}

void expect_config_eq(const MOE::Config& actual, const MOE::Config& expected) {
    EXPECT_EQ(actual.expert_type, expected.expert_type);
    EXPECT_EQ(actual.activation_type, expected.activation_type);
    EXPECT_FLOAT_EQ(actual.expert_alpha, expected.expert_alpha);
    EXPECT_FLOAT_EQ(actual.expert_beta, expected.expert_beta);
    EXPECT_EQ(actual.gate_idx, expected.gate_idx);
}

ov::Tensor make_weights_tensor(const std::string& bin) {
    ov::Tensor weights(ov::element::u8, ov::Shape{bin.size()});
    std::memcpy(weights.data(), bin.data(), bin.size());
    return weights;
}

}  // namespace

// Regression test for a bug where MOECompressed::m_config (its own Config, returned by get_config()) shadowed
// rather than shared MOE's base Config. XML deserialization only ever populated the base copy (via
// MOECompressed::visit_attributes() delegating to the non-virtual MOE::visit_attributes()), so
// MOECompressed::get_config() silently returned default values (e.g. expert_type/activation_type == 0) for any
// node loaded from an IR, and any later clone_with_new_inputs() (which copies from the own, broken copy) made
// the loss permanent.
TEST(MOECompressedConfigTest, SurvivesSerializeDeserializeRoundTrip) {
    MOECompressed::Config original_config;
    auto model = build_moe_compressed_model(original_config);

    std::stringstream xml;
    std::stringstream bin;
    ov::pass::Serialize(xml, bin).run_on_model(model);

    ov::Core core;
    core.add_extension(ov::OpExtension<MOE>());
    core.add_extension(ov::OpExtension<MOECompressed>());
    auto loaded_model = core.read_model(xml.str(), make_weights_tensor(bin.str()));

    auto loaded_moe = find_moe_compressed(loaded_model);
    ASSERT_NE(loaded_moe, nullptr);

    expect_config_eq(loaded_moe->get_config(), original_config);
    // MOE::get_config() is a different, non-virtual override reading a separate storage; it must agree too.
    expect_config_eq(static_cast<const MOE&>(*loaded_moe).get_config(), original_config);
}

TEST(MOECompressedConfigTest, SurvivesCloneAfterDeserialize) {
    MOECompressed::Config original_config;
    auto model = build_moe_compressed_model(original_config);

    std::stringstream xml;
    std::stringstream bin;
    ov::pass::Serialize(xml, bin).run_on_model(model);

    ov::Core core;
    core.add_extension(ov::OpExtension<MOE>());
    core.add_extension(ov::OpExtension<MOECompressed>());
    auto loaded_model = core.read_model(xml.str(), make_weights_tensor(bin.str()));

    auto loaded_moe = find_moe_compressed(loaded_model);
    ASSERT_NE(loaded_moe, nullptr);

    auto cloned = loaded_moe->clone_with_new_inputs(loaded_moe->input_values());
    auto cloned_moe = std::dynamic_pointer_cast<MOECompressed>(cloned);
    ASSERT_NE(cloned_moe, nullptr);

    expect_config_eq(cloned_moe->get_config(), original_config);
    expect_config_eq(static_cast<const MOE&>(*cloned_moe).get_config(), original_config);
}
