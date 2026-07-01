// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_compiled_model_utils.hpp"
#include "npuw_transformations/optimize_value_tensors.hpp"
#include "openvino/op/ops.hpp"
#include "partitioning/partitioning.hpp"

/*
 * conditional compilation that can be used during test regression debug
 * #define ANALYZE_TEST
 * which in turn will dump subgraphs after partitioning
 */

namespace npuw_utest {
using NodePtr = std::shared_ptr<ov::Node>;
}

enum class NetworkKind {
    MHA,  // Multi-Head Attention (e.g., llama2) - no broadcast
    GQA   // Grouped Query Attention (e.g., llama3, phi3, mistral, GPT-OSS) - with broadcast
};

typedef std::tuple<ov::Shape,
                   bool,  // withConvert
                   bool,  // withTranspose - without transpose node - matcher shouldnt detect subgraph, easy way to
                          // negative case
                   bool,  // withSDPA - should SDPA layer present or be already unrolled or simplified
                   bool,  // use high precision on attention_mask input
                   bool,  // withSink - SDPA with 6th input (sink) for GPT-OSS pattern
                   NetworkKind>
    OptimizeVTTestParamsTuple;

struct OptimizeVTTestParams {
#define _AT(idx) std::tuple_element<idx, OptimizeVTTestParamsTuple>::type

    _AT(0) inputShape;
    _AT(1) withConvert;
    _AT(2) withTranspose;
    _AT(3) withSDPA;
    _AT(4) withHpAttenMask;
    _AT(5) withSink;
    _AT(6) kind;
#undef _AT

    OptimizeVTTestParams(const OptimizeVTTestParamsTuple& tup) {
        std::tie(inputShape, withConvert, withTranspose, withSDPA, withHpAttenMask, withSink, kind) = tup;
    }
};

// based on ConcatWithDifferentChildrenTransformation
class TransposeVTTest : public testing::WithParamInterface<OptimizeVTTestParamsTuple>, public ov::test::TestsCommon {
public:
    void Validate() const {
        auto test = OptimizeVTTestParams{GetParam()};

        auto isValidSubgraph = test.withTranspose;
        ASSERT_EQ(isValidSubgraph, ov::npuw::util::OptimizeValueTensors(test.withHpAttenMask).run_on_model(model));

        // std::shared_ptr<ov::Model> model = ...;  // your model

        auto test_case_name = getTestCaseName(testing::TestParamInfo<OptimizeVTTestParamsTuple>{GetParam(), 0});
        std::string xml_path = test_case_name + ".xml";
        std::string bin_path = test_case_name + ".bin";

#ifdef ANALYZE_TEST
        // Save the model
        ov::pass::Serialize serialize_pass(xml_path, bin_path);
        serialize_pass.run_on_model(model);
#endif

        // validation of High Precision attention mask - implies enabling SDPA layer to be unrolled,
        // and also specific FP16 activation transformation in partitioning
        // Note: When withSink=true, standard OpenVINO SDPA decomposition is used which doesn't support HP
        if (test.withSDPA && !test.withSink) {
            std::shared_ptr<::intel_npu::OptionsDesc> options_desc;

            auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
            auto cfg = ::intel_npu::Config(opt_desc);
            ::intel_npu::registerNPUWOptions(*opt_desc);
            std::map<std::string, std::string> cfg_map = {
                {"NPUW_F16IC", "YES"}};  //, {"NPUW_ONLINE_PIPELINE", "NONE"}};
            cfg.update(cfg_map);

            ov::npuw::Partitioning partitioning;
            ASSERT_NO_THROW(partitioning = ov::npuw::getPartitioning(model, cfg));

            // input to add is 32b via convert / or via 32b parameter
            bool bAttentionMaskVerified = false;
            bool bAttentionMaskResultVerified = false;

            auto get_rank = [](const ov::Shape& sh) {
                return std::count_if(sh.begin(), sh.end(), [](size_t dim) {
                    return dim != 1;
                });
            };

            for (auto& subgraph : partitioning.subgraphs) {
                auto partitioned_model =
                    std::make_shared<ov::Model>(subgraph._results, subgraph._sinks, subgraph._parameters, "m1");
#ifdef ANALYZE_TEST
                auto test_case_name = getTestCaseName(testing::TestParamInfo<OptimizeVTTestParamsTuple>{GetParam(), 0});
                std::string xml_path = test_case_name + "_" + std::to_string(idx) + "_partitioned.xml";
                std::string bin_path = test_case_name + "_" + std::to_string(idx) + "_partitioned.bin";

                // Save the model
                ov::pass::Serialize serialize_pass(xml_path, bin_path);
                serialize_pass.run_on_model(partitioned_model);
#endif

                for (auto op : partitioned_model->get_ordered_ops()) {
                    // case when only 1 add and 1 negate layer in whole subgraph
                    if (ov::is_type<ov::op::v1::Add>(op)) {
                        ASSERT_FALSE(bAttentionMaskVerified);
                        // check rt_info
                        // should not be any convert operation for this add
                        // assume in lhs we have a mask
                        auto lhs = op->get_input_node_ptr(0);
                        auto rhs = op->get_input_node_ptr(1);

                        ASSERT_EQ(lhs->get_output_size(), 1);
                        ASSERT_EQ(rhs->get_output_size(), 1);

                        ASSERT_NE(lhs, nullptr)
                            << "Add layer " << op->get_friendly_name() << " need to have two inputs";
                        ASSERT_NE(rhs, nullptr)
                            << "Add layer " << op->get_friendly_name() << " need to have two inputs";

                        if (get_rank(lhs->get_output_shape(0)) != 2) {
                            ASSERT_EQ(get_rank(rhs->get_output_shape(0)), 2)
                                << "Add layer " << op->get_friendly_name() << " should have 2D input, but was{"
                                << lhs->get_output_shape(0) << " , " << rhs->get_output_shape(0) << " }";
                            std::swap(lhs, rhs);
                        }

                        if (test.withHpAttenMask) {
                            static constexpr char err[] = "in case of high precision, attention_mask has to accept "
                                                          "input in fp32 without convert layer";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Parameter>(lhs))
                                << err << ", actual type: " << lhs->get_type_name();
                            ASSERT_EQ(ov::as_type<ov::op::v0::Parameter>(lhs)->get_element_type(), ov::element::f32)
                                << err;
                        } else {
                            static constexpr char err[] = "in case of fp16 precision, attention_mask should have "
                                                          "preceding convert layer from fp16 to fp32";
                            // input has convert from fp16 to fp32
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(lhs))
                                << err << ", actual type: " << lhs->get_type_name();
                            ;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_destination_type(), ov::element::f32)
                                << err;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_input_element_type(0),
                                      ov::element::f16)
                                << err;
                        }

                        // should be only one add as atention_mask
                        bAttentionMaskVerified = true;
                    }

                    if (ov::is_type<ov::op::v0::Negative>(op)) {
                        ASSERT_FALSE(bAttentionMaskResultVerified);
                        // check rt_info
                        // should not be any convert operation after this negate
                        ASSERT_EQ(op->outputs().size(), 1);
                        auto result_s = op->output(0).get_target_inputs();

                        ASSERT_EQ(result_s.size(), 1);
                        auto result = result_s.begin()->get_node();

                        if (test.withHpAttenMask) {
                            static constexpr char err[] =
                                "in case of high precision, attention_mask producer need to be in fp32";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Result>(result))
                                << err << ", expected type Result, actual type: " << result->get_type_name();
                            ASSERT_EQ(ov::as_type<ov::op::v0::Result>(result)->get_element_type(), ov::element::f32)
                                << err;
                        } else {
                            static constexpr char err[] = "in case of fp16 precision, attention_mask producer should "
                                                          "have convert layer from fp32 to fp16";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(result))
                                << err << ", actual type: " << result->get_type_name();
                            ;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_destination_type(),
                                      ov::element::f16)
                                << err;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_input_element_type(0),
                                      ov::element::f32)
                                << err;
                        }

                        // should be only one add as atention_mask
                        bAttentionMaskResultVerified = true;
                    }
                }
            }

            ASSERT_TRUE(bAttentionMaskVerified)
                << "no attention mask node not detected after applying optimize_value_tensors + run getPartitioning";
            ASSERT_TRUE(bAttentionMaskResultVerified) << "no attention mask producer node detected after applying "
                                                         "optimize_value_tensors + run getPartitioning";
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<OptimizeVTTestParamsTuple>& obj) {
        auto test = OptimizeVTTestParams{obj.param};

        std::ostringstream result;
        result << "npuw_llm_pipeline_" << test.inputShape << "_" << (test.kind == NetworkKind::MHA ? "MHA" : "GQA")
               << (test.withConvert ? "_with_convert" : "") << (test.withSDPA ? "_SDPA" : "")
               << (test.withSink ? "_Sink" : "") << (test.withHpAttenMask ? "_HP" : "")
               << (!test.withTranspose ? "_NEGATIVE" : "");
        return result.str();
    }

protected:
    void SetUp() override {
        model = CreateModel();
    }

    std::shared_ptr<ov::Model> CreateModel() {
        const auto test = OptimizeVTTestParams{GetParam()};

        auto create_shape_constant = [](const std::vector<int64_t>& const_data, const std::string& name) {
            auto pattern =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{const_data.size()}, const_data);
            pattern->set_friendly_name("unsqueese_pattern");
            return pattern;
        };

        // in case of non broadcast number of input channels significantly smaller
        auto numChannels = (test.kind == NetworkKind::MHA) ? 32 : 8;
        auto input_shape = test.inputShape;
        auto input_2 = static_cast<int>(test.inputShape[2]);
        auto input_3 = static_cast<int>(test.inputShape[3]);

        input_shape.at(1) = numChannels;

        // ov::Model with only a transpose node
        auto param = std::make_shared<ov::op::v0::Parameter>(test.withConvert ? ov::element::f16 : ov::element::f32,
                                                             input_shape);
        param->set_friendly_name("past_key_value");

        std::shared_ptr<ov::Node> convert =
            test.withConvert
                ? std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, ov::element::f32))
                : std::static_pointer_cast<ov::Node>(param);
        if (test.withConvert) {
            convert->set_friendly_name("convert");
        }

        // todo parametrise optional reshape
        auto param2 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[3] * numChannels});
        param2->set_friendly_name("new_token");

        auto reshape_pattern = create_shape_constant({0, 0, numChannels, input_3}, "reshape_pattern");
        auto transpose_pattern = create_shape_constant({1, numChannels, 1, input_3}, "transposed_pattern");

        auto reshape = std::make_shared<ov::op::v1::Reshape>(param2,
                                                             test.withTranspose ? reshape_pattern : transpose_pattern,
                                                             true);
        reshape->set_friendly_name("reshape");

        std::shared_ptr<ov::Node> transpose_or_reshape;

        if (test.withTranspose) {
            auto constOrder = create_shape_constant({0, 2, 1, 3}, "const_order");
            auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, constOrder);
            transpose->set_friendly_name("transpose");
            transpose_or_reshape = transpose;
        } else {
            transpose_or_reshape = reshape;
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convert, transpose_or_reshape}, -2);
        concat->set_friendly_name("concat");

        std::shared_ptr<ov::Node> concat_or_reshape = concat;

        if (test.kind == NetworkKind::GQA) {
            auto unsqueeze_pattern = create_shape_constant({2}, "unsqueese_pattern");
            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, unsqueeze_pattern);
            unsqueeze->set_friendly_name("unsqueeze");

            auto broadcast_pattern = create_shape_constant({1, 8, 4, input_2 + 1, input_3}, "broadcast_pattern");
            // TODO: v1::Broadcast not working
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze,
                                                                     broadcast_pattern,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
            broadcast->set_friendly_name("broadcast");

            auto reshape_pattern2 = create_shape_constant({0, 32, -1, input_3}, "reshape_pattern2");
            auto reshape2 = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_pattern2, true);
            reshape2->set_friendly_name("reshape2");

            concat_or_reshape = reshape2;
        }

        if (test.withSDPA) {
            auto mask_input =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 1, input_shape[2] + 1, input_shape[2] + 1});
            mask_input->set_friendly_name("mask_input");

            auto mask_input_1 = std::make_shared<ov::op::v0::Negative>(mask_input);
            mask_input_1->set_friendly_name("mask_input_1");

            auto k_input =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 32, input_shape[2] + 1, input_shape[3]});
            k_input->set_friendly_name("k_input");

            auto q_input =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        ov::Shape{1, 32, input_shape[2] + 1, input_shape[3]});
            q_input->set_friendly_name("q_input");

            auto scale_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});

            std::shared_ptr<ov::Node> sdpa;
            ov::ParameterVector params = {param, param2, mask_input, k_input, q_input};

            // SDPA with sink (6 inputs) for GPT-OSS pattern
            if (test.withSink) {
                auto sink = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 1, 1});
                sink->set_friendly_name("sink");
                params.push_back(sink);

                sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_input,
                                                                                k_input,
                                                                                concat_or_reshape,
                                                                                mask_input_1,
                                                                                scale_node,
                                                                                sink,
                                                                                false);
            } else {
                sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_input,
                                                                                k_input,
                                                                                concat_or_reshape,
                                                                                mask_input_1,
                                                                                scale_node,
                                                                                false);
            }

            sdpa->set_friendly_name("sdpa");
            auto result = std::make_shared<ov::op::v0::Result>(sdpa);

            result->set_friendly_name("res");
            return std::make_shared<ov::Model>(ov::ResultVector{result}, params);

        } else {
            auto param3 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 1, input_shape[2] + 1});
            param3->set_friendly_name("param3");

            // TODO: what if v1 softmax???
            auto softmax = std::make_shared<ov::op::v8::Softmax>(param3, -2);
            softmax->set_friendly_name("softmax");

            // entry point matmul for matcher
            auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, concat_or_reshape);
            matmul->set_friendly_name("matmul");

            auto result = std::make_shared<ov::op::v0::Result>(matmul);
            result->set_friendly_name("res");
            return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, param2, param3});
        }
    }

    std::shared_ptr<ov::Model> model;
};

TEST_P(TransposeVTTest, smoke_Run_MatchAndTransposeVT) {
    Validate();
}

namespace {
// eliminate direct shape dependency to match llama2, as in test and in optimize function
const std::vector<ov::Shape> input_shapes{{1, 0, 1151, 128}, {1, 0, 1141, 64}};

const std::vector<bool> withTranspose{true, false};

const std::vector<bool> withBroadCast{true, false};

const std::vector<bool> withSDPA{true, false};

const std::vector<bool> withHpAttenMask{true, false};

const std::vector<bool> withSink{true, false};

const std::vector<NetworkKind> networkKind = {
    NetworkKind::MHA,  // Multi-Head Attention
    NetworkKind::GQA   // Grouped Query Attention
};

INSTANTIATE_TEST_SUITE_P(smoke_Run_MatchAndTransposeVT,
                         TransposeVTTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(withTranspose),
                                            ::testing::ValuesIn(withBroadCast),
                                            ::testing::ValuesIn(withSDPA),
                                            ::testing::ValuesIn(withHpAttenMask),
                                            ::testing::ValuesIn(withSink),
                                            ::testing::ValuesIn(networkKind)),
                         TransposeVTTest::getTestCaseName);

}  // namespace

// Regression test for shared V-concat (e.g. Gemma4 local+global attention).
// When the same past_value->concat is consumed by two matmul branches, GraphRewrite
// fires the MHA matcher once per matmul.  The shared mutations (param shape swap,
// transpose order, concat axis) must be applied exactly once; transpose_b is
// per-branch and must be set on both.
//
// Model topology:
//
//   param[1,4,8,64]   param2[1,1,256]
//        |                  |
//        |           reshape[1,1,4,64]
//        |                  |
//        |           transpose{0,2,1,3}[1,4,1,64]
//        +-------concat(axis=2)[1,4,9,64]--------+
//                      |                          |
//              softmax1->matmul1          softmax2->matmul2
//
TEST(TransposeVTSharedConcatTest, smoke_SharedConcat_MHA_MultipleUsers) {
    const size_t B = 1, H = 4, S = 8, D = 64;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, S, D});
    param->set_friendly_name("past_value");

    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, 1, H * D});
    param2->set_friendly_name("new_token");

    auto reshape_cst = ov::op::v0::Constant::create(ov::element::i64,
                                                    ov::Shape{4},
                                                    std::vector<int64_t>{0, 0, (int64_t)H, (int64_t)D});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(param2, reshape_cst, true);

    auto tr_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, tr_order);
    transpose->set_friendly_name("v_transpose");

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{param, transpose}, 2);
    concat->set_friendly_name("v_concat");

    auto attn1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, 1, S + 1});
    attn1->set_friendly_name("attn_scores_1");
    auto matmul1 = std::make_shared<ov::op::v0::MatMul>(std::make_shared<ov::op::v8::Softmax>(attn1, -1), concat);
    matmul1->set_friendly_name("matmul1");

    auto attn2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{B, H, 1, S + 1});
    attn2->set_friendly_name("attn_scores_2");
    auto matmul2 = std::make_shared<ov::op::v0::MatMul>(std::make_shared<ov::op::v8::Softmax>(attn2, -1), concat);
    matmul2->set_friendly_name("matmul2");

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(matmul1), std::make_shared<ov::op::v0::Result>(matmul2)},
        ov::ParameterVector{param, param2, attn1, attn2});

    bool transposed = false;
    ASSERT_NO_THROW(transposed = ov::npuw::util::OptimizeValueTensors(false).run_on_model(model));
    ASSERT_TRUE(transposed);

    // param swapped exactly once: [B,H,S,D] -> [B,H,D,S]
    const auto& ps = param->get_partial_shape();
    ASSERT_EQ(ps[2].get_length(), (int64_t)D);
    ASSERT_EQ(ps[3].get_length(), (int64_t)S);

    ASSERT_EQ(concat->get_axis(), 3);
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v0::MatMul>(matmul1)->get_transpose_b());
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v0::MatMul>(matmul2)->get_transpose_b());
}
