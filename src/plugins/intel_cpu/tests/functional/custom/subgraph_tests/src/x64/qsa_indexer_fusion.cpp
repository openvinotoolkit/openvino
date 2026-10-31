// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "custom/single_layer_tests/classes/qsa_indexer.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

namespace {

using namespace ov::op;

constexpr size_t kBatch = 2;
constexpr size_t kQueryHeads = 2;
constexpr size_t kHeadDim = 16;
constexpr size_t kRotaryDim = 8;
constexpr size_t kBlockTopk = 2;
constexpr size_t kOracleSummaryCap = 4;
constexpr float kEpsilon = 1e-6F;

// kWorkedExample: key_block_size 4, a prefill call (L=10, key_length=[10,7], row 1 left-padded by 3) followed by a
//                 decode call (L=1, key_length=[11,8]) that completes row 1's second block.
// kDensePrefill: key_block_size 2, a single call with key_length[b] == L for every row.
// kDensePrefillSharedRope: as kDensePrefill, but the RoPE tables have a batch of 1 (same positions in every row).
enum class Scenario { kWorkedExample, kDensePrefill, kDensePrefillSharedRope };

// scenario, RMSNorm of the query applied after the [N, L, H, D] -> [N, H, L, D] transpose instead of before it,
// precision (the element type of the model's floating-point inputs, constants and outputs, and the inference precision)
using QSAIndexerFusionParams = std::tuple<Scenario, bool, element::Type>;

struct CallSpec {
    int64_t length = 0;
    std::vector<int32_t> key_length;
};

std::string scenario_name(Scenario scenario) {
    switch (scenario) {
    case Scenario::kWorkedExample:
        return "WorkedExample";
    case Scenario::kDensePrefill:
        return "DensePrefill";
    case Scenario::kDensePrefillSharedRope:
        return "DensePrefillSharedRope";
    }
    return {};
}

std::shared_ptr<v0::Constant> scalar(const element::Type& type, float value) {
    return v0::Constant::create(type, Shape{}, {value});
}

std::shared_ptr<v0::Constant> i64_const(const std::vector<int64_t>& values) {
    return v0::Constant::create(element::i64, Shape{values.size()}, values);
}

// x * 1 / sqrt(mean(x^2, -1) + eps) * gamma
Output<Node> make_rms_norm(const Output<Node>& x, const Output<Node>& gamma) {
    const auto type = x.get_element_type();
    auto square = std::make_shared<v1::Power>(x, scalar(type, 2.0F));
    auto mean = std::make_shared<v1::ReduceMean>(square, v0::Constant::create(element::i32, Shape{1}, {-1}), true);
    auto add_eps = std::make_shared<v1::Add>(mean, scalar(type, kEpsilon));
    auto rsqrt = std::make_shared<v1::Divide>(scalar(type, 1.0F), std::make_shared<v0::Sqrt>(add_eps));
    auto norm = std::make_shared<v1::Multiply>(x, rsqrt);
    return std::make_shared<v1::Multiply>(norm, gamma);
}

Output<Node> make_slice(const Output<Node>& x, int64_t start, int64_t stop) {
    return std::make_shared<v8::Slice>(x, i64_const({start}), i64_const({stop}), i64_const({1}), i64_const({3}));
}

// Rotate-half RoPE of the first Dr channels of x [N, H, L, D] with cos / sin [N, 1, L, Dr], as in HF Transformers:
//   y = x_rot * cos + concat(-x_rot[Dr / 2:], x_rot[:Dr / 2]) * sin, followed by the untouched rest of the channels.
Output<Node> make_partial_rope(const Output<Node>& x, const Output<Node>& cos, const Output<Node>& sin) {
    constexpr int64_t int32_max = std::numeric_limits<int32_t>::max();
    constexpr int64_t rotary_dim = kRotaryDim;
    auto x_rot = make_slice(x, 0, rotary_dim);
    auto x_pass = make_slice(x, rotary_dim, int32_max);
    auto x_first_half = make_slice(x_rot, 0, rotary_dim / 2);
    auto x_second_half = make_slice(x_rot, rotary_dim / 2, int32_max);
    auto x_second_half_neg = std::make_shared<v1::Multiply>(x_second_half, scalar(x.get_element_type(), -1.0F));
    auto rotate_half = std::make_shared<v0::Concat>(OutputVector{x_second_half_neg, x_first_half}, -1);
    auto rotated = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(x_rot, cos),
                                             std::make_shared<v1::Multiply>(rotate_half, sin));
    return std::make_shared<v0::Concat>(OutputVector{rotated, x_pass}, -1);
}

}  // namespace

// Builds the public QSA index selection graph (regular query RMSNorm + RoPE, QSAPrepare-17,
// SparseAttentionIndexer-17), runs it on CPU, where QSAIndexerFusion has to turn it into one QSAIndexer node, and
// compares everything it returns with the host reference of the QSAIndexer kernel.
class QSAIndexerFusionCPUTest : public testing::WithParamInterface<QSAIndexerFusionParams>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QSAIndexerFusionParams>& obj) {
        const auto& [scenario, rms_after_transpose, precision] = obj.param;
        return "Scenario=" + scenario_name(scenario) + "_RMSAfterTranspose=" + std::to_string(rms_after_transpose) +
               "_Precision=" + precision.to_string();
    }

protected:
    void SetUp() override {
        const auto& [scenario, rms_after_transpose, precision] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration[ov::hint::inference_precision.name()] = precision;
        m_precision = precision;
        // The kernel keeps its caches in this precision, the oracle computes in f32.
        if (precision == element::bf16) {
            m_abs_tolerance = m_rel_tolerance = 5e-2F;
        } else if (precision == element::f16) {
            m_abs_tolerance = m_rel_tolerance = 1e-2F;
        }
        m_scenario = scenario;

        if (scenario == Scenario::kWorkedExample) {
            m_key_block_size = 4;
            m_calls = {CallSpec{10, {10, 7}}, CallSpec{1, {11, 8}}};
        } else {
            m_key_block_size = 2;
            m_calls = {CallSpec{7, std::vector<int32_t>(kBatch, 7)}};
        }
        const auto key_block_size = static_cast<int64_t>(m_key_block_size);
        const auto token_budget = static_cast<int64_t>(m_key_block_size * kBlockTopk);
        const auto batch = static_cast<int64_t>(kBatch);
        const auto head_dim = static_cast<int64_t>(kHeadDim);
        const auto rotary_dim = static_cast<int64_t>(kRotaryDim);
        const auto rope_batch = scenario == Scenario::kDensePrefillSharedRope ? int64_t{1} : batch;

        auto q = std::make_shared<v0::Parameter>(m_precision,
                                                 PartialShape{batch, -1, static_cast<int64_t>(kQueryHeads), head_dim});
        auto raw_keys = std::make_shared<v0::Parameter>(m_precision, PartialShape{batch, -1, head_dim});
        auto rope_cos = std::make_shared<v0::Parameter>(m_precision, PartialShape{rope_batch, -1, rotary_dim});
        auto rope_sin = std::make_shared<v0::Parameter>(m_precision, PartialShape{rope_batch, -1, rotary_dim});
        auto key_length = std::make_shared<v0::Parameter>(element::i32, PartialShape{batch});
        auto past_blocks = std::make_shared<v0::Parameter>(m_precision, PartialShape{batch, -1, head_dim});
        auto past_raw_keys =
            std::make_shared<v0::Parameter>(m_precision, PartialShape{batch, key_block_size - 1, head_dim});
        auto past_rope = std::make_shared<v0::Parameter>(m_precision, PartialShape{batch, 2 * rotary_dim});
        m_params = {q, raw_keys, rope_cos, rope_sin, key_length, past_blocks, past_raw_keys, past_rope};

        m_q_norm_weight =
            utils::create_and_fill_tensor(m_precision, Shape{kHeadDim}, utils::InputGenerateData(0.5, 1000, 1000, 6));
        m_k_norm_weight =
            utils::create_and_fill_tensor(m_precision, Shape{kHeadDim}, utils::InputGenerateData(0.5, 1000, 1000, 7));
        auto q_norm_weight = std::make_shared<v0::Constant>(m_q_norm_weight);
        auto k_norm_weight = std::make_shared<v0::Constant>(m_k_norm_weight);

        // ---- index query: RMSNorm + RoPE, regular graph operations ----
        const auto order = i64_const({0, 2, 1, 3});
        Output<Node> index_query;
        if (rms_after_transpose) {
            index_query = std::make_shared<v1::Transpose>(q, order);
            index_query = make_rms_norm(index_query, q_norm_weight);
        } else {
            index_query = make_rms_norm(q, q_norm_weight);
            index_query = std::make_shared<v1::Transpose>(index_query, order);
        }
        auto cos_4d = std::make_shared<v0::Unsqueeze>(rope_cos, i64_const({1}));
        auto sin_4d = std::make_shared<v0::Unsqueeze>(rope_sin, i64_const({1}));
        index_query = make_partial_rope(index_query, cos_4d, sin_4d);

        // ---- QSAPrepare-17 + SparseAttentionIndexer-17 ----
        auto prepare = std::make_shared<v17::QSAPrepare>(raw_keys,
                                                         past_blocks,
                                                         key_length,
                                                         rope_cos,
                                                         rope_sin,
                                                         k_norm_weight,
                                                         past_raw_keys,
                                                         past_rope,
                                                         key_block_size,
                                                         kEpsilon);
        auto indexer = std::make_shared<v17::SparseAttentionIndexer>(index_query,
                                                                     prepare->output(0),
                                                                     key_length,
                                                                     key_block_size,
                                                                     token_budget);

        function = std::make_shared<Model>(ResultVector{std::make_shared<v0::Result>(indexer->output(0)),
                                                        std::make_shared<v0::Result>(indexer->output(1)),
                                                        std::make_shared<v0::Result>(prepare->output(0)),
                                                        std::make_shared<v0::Result>(prepare->output(1)),
                                                        std::make_shared<v0::Result>(prepare->output(2))},
                                           m_params,
                                           "QSAIndexerFusion");
    }

    void run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        const auto inference_precision = std::get<2>(GetParam());
        if (inference_precision == element::bf16 && !(with_cpu_x86_bfloat16() || with_cpu_x86_avx2_vnni_2())) {
            GTEST_SKIP() << "bf16 inference is not supported on this CPU";
        }
        if (inference_precision == element::f16 && !(with_cpu_x86_avx512_core_fp16() || with_cpu_x86_avx2_vnni_2())) {
            GTEST_SKIP() << "f16 inference is not supported on this CPU";
        }
        compiledModel = core->compile_model(function, targetDevice, configuration);
        check_fused();
        auto infer_request = compiledModel.create_infer_request();

        // Caller-owned state of the public ops, fed back from their outputs; the oracle keeps its own copy.
        const auto head_dim = kHeadDim;
        Tensor past_blocks(m_precision, Shape{kBatch, 0, head_dim});
        Tensor past_raw_keys = make_zero_tensor(m_precision, Shape{kBatch, m_key_block_size - 1, head_dim});
        Tensor past_rope = make_zero_tensor(m_precision, Shape{kBatch, 2 * kRotaryDim});

        QSAIndexerOracle oracle;
        oracle.h_iq = kQueryHeads;
        oracle.h_ik = 1;
        oracle.h_sel = 1;
        oracle.d_idx = kHeadDim;
        oracle.rotary_dim = kRotaryDim;
        oracle.r = static_cast<int64_t>(m_key_block_size);
        oracle.block_topk = kBlockTopk;
        oracle.eps = kEpsilon;
        oracle.scale = 1.0F;
        // the selection is an output of the model: the fused op returns it in the order of SparseAttentionIndexer-17
        oracle.order_by_score = true;
        oracle.summary_cap = kOracleSummaryCap;
        oracle.summary_cache.assign(kBatch * kOracleSummaryCap * head_dim, 0.0F);
        oracle.past_raw_k.assign(kBatch * (m_key_block_size - 1) * head_dim, 0.0F);
        oracle.past_rope.assign(kBatch * 2 * kRotaryDim, 0.0F);

        for (size_t call_idx = 0; call_idx < m_calls.size(); ++call_idx) {
            const auto& call = m_calls[call_idx];
            const auto length = static_cast<size_t>(call.length);
            const auto fill = [&](const Shape& shape, size_t slot) {
                return utils::create_and_fill_tensor(
                    m_precision,
                    shape,
                    utils::InputGenerateData(-1.0, 2000, 1000, static_cast<int32_t>(10 + call_idx * 4 + slot)));
            };
            Tensor q = fill(Shape{kBatch, length, kQueryHeads, head_dim}, 1);
            Tensor raw_keys = fill(Shape{kBatch, length, head_dim}, 2);

            // cos / sin must be genuine (cos, sin) pairs of the absolute positions of the new tokens.
            const size_t rope_batch = m_scenario == Scenario::kDensePrefillSharedRope ? 1 : kBatch;
            std::vector<float> cos_data(rope_batch * length * kRotaryDim);
            std::vector<float> sin_data(rope_batch * length * kRotaryDim);
            for (size_t b = 0; b < rope_batch; ++b) {
                for (size_t l = 0; l < length; ++l) {
                    const int64_t position = call.key_length[b] - call.length + static_cast<int64_t>(l);
                    for (size_t j = 0; j < kRotaryDim; ++j) {
                        const float theta =
                            static_cast<float>(position) * 0.1F + static_cast<float>(j % (kRotaryDim / 2)) * 0.05F;
                        cos_data[(b * length + l) * kRotaryDim + j] = std::cos(theta);
                        sin_data[(b * length + l) * kRotaryDim + j] = std::sin(theta);
                    }
                }
            }
            Tensor rope_cos(m_precision, Shape{rope_batch, length, kRotaryDim});
            Tensor rope_sin(m_precision, Shape{rope_batch, length, kRotaryDim});
            write_f32_to_tensor(rope_cos, cos_data);
            write_f32_to_tensor(rope_sin, sin_data);
            // The oracle sees the tables as the model does, rounded to the model type.
            cos_data = read_tensor_to_f32(rope_cos);
            sin_data = read_tensor_to_f32(rope_sin);

            Tensor key_length(element::i32, Shape{kBatch});
            std::copy(call.key_length.begin(), call.key_length.end(), key_length.data<int32_t>());

            const std::vector<Tensor> inputs =
                {q, raw_keys, rope_cos, rope_sin, key_length, past_blocks, past_raw_keys, past_rope};
            for (size_t i = 0; i < inputs.size(); ++i) {
                infer_request.set_input_tensor(i, inputs[i]);
            }
            infer_request.infer();
            std::vector<Tensor> outputs;
            for (size_t i = 0; i < compiledModel.outputs().size(); ++i) {
                const auto& output = infer_request.get_output_tensor(i);
                Tensor copy(output.get_element_type(), output.get_shape());
                output.copy_to(copy);
                outputs.push_back(std::move(copy));
            }

            // The oracle takes the QSAIndexer layout: [N, L, H, D] queries and [N, L, H_ik, D] keys, [N, L, Dr] tables.
            std::vector<float> oracle_cos(kBatch * length * kRotaryDim);
            std::vector<float> oracle_sin(kBatch * length * kRotaryDim);
            for (size_t b = 0; b < kBatch; ++b) {
                const size_t src_b = rope_batch == 1 ? 0 : b;
                std::copy_n(cos_data.begin() + src_b * length * kRotaryDim,
                            length * kRotaryDim,
                            oracle_cos.begin() + b * length * kRotaryDim);
                std::copy_n(sin_data.begin() + src_b * length * kRotaryDim,
                            length * kRotaryDim,
                            oracle_sin.begin() + b * length * kRotaryDim);
            }
            const auto expected = oracle.run_call(kBatch,
                                                  call.length,
                                                  call.key_length,
                                                  read_tensor_to_f32(q),
                                                  read_tensor_to_f32(raw_keys),
                                                  oracle_cos,
                                                  oracle_sin,
                                                  read_tensor_to_f32(m_q_norm_weight),
                                                  read_tensor_to_f32(m_k_norm_weight));

            compare_call(call, expected, oracle, outputs);

            past_blocks = outputs[2];
            past_raw_keys = outputs[3];
            past_rope = outputs[4];
        }
    }

private:
    // QSAPrepare and SparseAttentionIndexer have to be gone, replaced by exactly one QSAIndexer.
    void check_fused() const {
        size_t qsa_indexers = 0;
        for (const auto& node : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto layer_type = node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            EXPECT_NE(layer_type, "QSAPrepare");
            EXPECT_NE(layer_type, "SparseAttentionIndexer");
            qsa_indexers += layer_type == "QSAIndexer";
        }
        ASSERT_EQ(qsa_indexers, 1u);
    }

    void compare_call(const CallSpec& call,
                      const QSAIndexerOracle::Output& expected,
                      const QSAIndexerOracle& oracle,
                      const std::vector<Tensor>& outputs) const {
        const auto length = static_cast<size_t>(call.length);
        const auto r = static_cast<int64_t>(m_key_block_size);

        // sel_indices [N, 1, L, block_topk], sel_count [N, 1, L]
        ASSERT_EQ(outputs[0].get_shape(), (Shape{kBatch, 1, length, kBlockTopk}));
        ASSERT_EQ(outputs[1].get_shape(), (Shape{kBatch, 1, length}));
        EXPECT_TRUE(std::equal(expected.sel_indices.begin(), expected.sel_indices.end(), outputs[0].data<int32_t>()))
            << "sel_indices differ";
        EXPECT_TRUE(std::equal(expected.sel_count.begin(), expected.sel_count.end(), outputs[1].data<int32_t>()))
            << "sel_count differ";

        // block_index_keys [N, longest_key_length / r, D]: only the complete blocks of every row are defined.
        const int64_t longest = *std::max_element(call.key_length.begin(), call.key_length.end());
        ASSERT_EQ(outputs[2].get_shape(), (Shape{kBatch, static_cast<size_t>(longest / r), kHeadDim}));
        const auto block_keys = read_tensor_to_f32(outputs[2]);
        const auto num_blocks = static_cast<size_t>(longest / r);
        for (size_t b = 0; b < kBatch; ++b) {
            for (size_t block = 0; block < static_cast<size_t>(call.key_length[b] / r); ++block) {
                for (size_t d = 0; d < kHeadDim; ++d) {
                    const float want = oracle.summary_cache[(b * kOracleSummaryCap + block) * kHeadDim + d];
                    const float got = block_keys[(b * num_blocks + block) * kHeadDim + d];
                    EXPECT_NEAR(got, want, m_abs_tolerance + m_rel_tolerance * std::abs(want))
                        << "block_index_keys[" << b << "][" << block << "][" << d << "]";
                }
            }
        }

        // past_incomplete_raw_keys [N, r - 1, D] and past_incomplete_rope [N, 2 * Dr]
        ASSERT_EQ(outputs[3].get_shape(), (Shape{kBatch, m_key_block_size - 1, kHeadDim}));
        ASSERT_EQ(outputs[4].get_shape(), (Shape{kBatch, 2 * kRotaryDim}));
        const auto past_raw_keys = read_tensor_to_f32(outputs[3]);
        const auto past_rope = read_tensor_to_f32(outputs[4]);
        for (size_t i = 0; i < oracle.past_raw_k.size(); ++i) {
            EXPECT_NEAR(past_raw_keys[i],
                        oracle.past_raw_k[i],
                        m_abs_tolerance + m_rel_tolerance * std::abs(oracle.past_raw_k[i]))
                << "past_incomplete_raw_keys[" << i << "]";
        }
        for (size_t i = 0; i < oracle.past_rope.size(); ++i) {
            EXPECT_NEAR(past_rope[i],
                        oracle.past_rope[i],
                        m_abs_tolerance + m_rel_tolerance * std::abs(oracle.past_rope[i]))
                << "past_incomplete_rope[" << i << "]";
        }
    }

    float m_abs_tolerance = 1e-4F;
    float m_rel_tolerance = 1e-3F;
    element::Type m_precision = element::f32;
    Scenario m_scenario = Scenario::kWorkedExample;
    size_t m_key_block_size = 4;
    std::vector<CallSpec> m_calls;
    ParameterVector m_params;
    Tensor m_q_norm_weight;
    Tensor m_k_norm_weight;
};

TEST_P(QSAIndexerFusionCPUTest, CompareWithRefs) {
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexerFusion_CPU,
                         QSAIndexerFusionCPUTest,
                         ::testing::Combine(::testing::Values(Scenario::kWorkedExample,
                                                              Scenario::kDensePrefill,
                                                              Scenario::kDensePrefillSharedRope),
                                            ::testing::Bool(),
                                            ::testing::Values(element::f32, element::bf16, element::f16)),
                         QSAIndexerFusionCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
