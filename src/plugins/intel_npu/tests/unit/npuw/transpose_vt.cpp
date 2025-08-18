// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include "openvino/op/ops.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_compiled_model_utils.hpp"
#include "partitioning/partitioning.hpp"
#include "intel_npu/config/config.hpp"

/*
* conditional compilation that can be used during test regression debug
* #define ANALYZE_TEST
* which in turn will dump subgraphs after partitioning
*/

namespace npuw_utest{
    using NodePtr = std::shared_ptr<ov::Node>;
}

enum class NetworkKind {
    llama2,
    llama3
};

typedef std::tuple <
    ov::Shape,
    bool,       // withConvert
    bool,       // withTranspose - without transpose node - matcher shouldnt detect subgraph, easy way to negative case
    bool,       // withSDPA - should SDPA layer present or be already unrolled or simplified
    bool,       // use high precision on attention_mask input
    NetworkKind
> OptimizeVTTestParamsTuple;

struct OptimizeVTTestParams {
    #define _AT(idx) std::tuple_element<idx, OptimizeVTTestParamsTuple>::type 

    _AT(0)  inputShape;
    _AT(1)  withConvert;
    _AT(2)  withTranspose;
    _AT(3)  withSDPA;
    _AT(4)  withHpAttenMask;
    _AT(5)  kind;
    #undef _AT

    OptimizeVTTestParams(const OptimizeVTTestParamsTuple& tup) {
        std::tie(inputShape, withConvert, withTranspose, withSDPA, withHpAttenMask, kind) = tup;
    }
};


// based on ConcatWithDifferentChildrenTransformation
class TransposeVTTest : public testing::WithParamInterface<OptimizeVTTestParamsTuple>,
                                     public ov::test::TestsCommon {
public:
    void Validate() const {
        auto test = OptimizeVTTestParams{GetParam()};
        
        auto isValidSubgraph  = test.withTranspose;
        ASSERT_EQ(isValidSubgraph, ov::npuw::util::optimize_value_tensors(model, test.withHpAttenMask));

        //std::shared_ptr<ov::Model> model = ...;  // your model

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
        if (test.withSDPA) {
            std::shared_ptr<::intel_npu::OptionsDesc> options_desc;

            auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
            auto cfg = ::intel_npu::Config(opt_desc);
            ::intel_npu::registerNPUWOptions(*opt_desc);
            std::map<std::string, std::string> cfg_map = {{"NPUW_F16IC", "YES"}};//, {"NPUW_ONLINE_PIPELINE", "NONE"}};
            cfg.update(cfg_map);

            ov::npuw::Partitioning partitioning;
            ASSERT_NO_THROW(partitioning = ov::npuw::getPartitioning(model, cfg));

            // input to add is 32b via convert / or via 32b parameter
            bool bAttentionMaskVerified = false;
            bool bAttentionMaskResultVerified = false;
            
            auto get_rank = [](const ov::Shape& sh) {
                return std::count_if(sh.begin(), sh.end(), [](size_t dim) {return dim != 1;});
            };

            for (auto & subgraph :  partitioning.subgraphs) {
                auto partitioned_model = std::make_shared<ov::Model>(subgraph._results,
                    subgraph._sinks,
                    subgraph._parameters,
                    "m1");
#ifdef ANALYZE_TEST
                auto test_case_name = getTestCaseName(testing::TestParamInfo<OptimizeVTTestParamsTuple>{GetParam(), 0});
                std::string xml_path = test_case_name + "_"+ std::to_string(idx) +"_partitioned.xml";
                std::string bin_path = test_case_name + "_"+ std::to_string(idx) +"_partitioned.bin";
        
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

                        ASSERT_NE(lhs, nullptr) << "Add layer " << op->get_friendly_name() << " need to have two inputs";
                        ASSERT_NE(rhs, nullptr) << "Add layer " << op->get_friendly_name() << " need to have two inputs";
                        
                        if (get_rank(lhs->get_output_shape(0)) != 2) {
                            ASSERT_EQ(get_rank(rhs->get_output_shape(0)), 2) 
                                << "Add layer " << op->get_friendly_name() << " should have 2D input, but was{" 
                                << lhs->get_output_shape(0) << " , " << rhs->get_output_shape(0) << " }";
                            std::swap(lhs, rhs);
                        }

                        if (test.withHpAttenMask) {
                            static constexpr char err[] = "in case of high precision, attention_mask has to accept input in fp32 without convert layer";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Parameter>(lhs)) << err << ", actual type: " << lhs->get_type_name();
                            ASSERT_EQ(ov::as_type<ov::op::v0::Parameter>(lhs)->get_element_type(), ov::element::f32) << err;
                        } else {
                            static constexpr char err[] = "in case of fp16 precision, attention_mask should have preceding convert layer from fp16 to fp32";
                            // input has convert from fp16 to fp32
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(lhs)) << err << ", actual type: " << lhs->get_type_name();;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_destination_type(), ov::element::f32) << err;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_input_element_type(0), ov::element::f16) << err;
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
                            static constexpr char err[] = "in case of high precision, attention_mask producer need to be in fp32";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Result>(result)) << err << ", expected type Result, actual type: " << result->get_type_name();
                            ASSERT_EQ(ov::as_type<ov::op::v0::Result>(result)->get_element_type(), ov::element::f32) << err;
                        } else {
                            static constexpr char err[] = "in case of fp16 precision, attention_mask producer should have convert layer from fp32 to fp16";
                            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(result)) << err << ", actual type: " << result->get_type_name();;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_destination_type(), ov::element::f16) << err;
                            ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_input_element_type(0), ov::element::f32) << err;
                        }

                        // should be only one add as atention_mask   
                        bAttentionMaskResultVerified = true;
                    }                    
                }
            }

            ASSERT_TRUE(bAttentionMaskVerified) << "no attention mask node not detected after applying optimize_value_tensors + run getPartitioning";
            ASSERT_TRUE(bAttentionMaskResultVerified) << "no attention mask producer node detected after applying optimize_value_tensors + run getPartitioning";
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<OptimizeVTTestParamsTuple> obj) {
        auto test = OptimizeVTTestParams{obj.param};

        std::ostringstream result;
        result << "npuw_llm_pipeline_" << test.inputShape << "_" 
               << (test.kind == NetworkKind::llama3 ?  "LLAMA3" : "LLAMA2") 
               << (test.withConvert ? "_with_convert" : "")
               << (test.withSDPA ? "_SDPA" : "")
               << (test.withHpAttenMask ? "_HP" : "")
               << (!test.withTranspose ? "_NEGATIVE" : "");
        return result.str();
    }    

protected:

    void SetUp() override {
        model = CreateModel();
    }

    std::shared_ptr<ov::Model> CreateModel() {

        const auto test = OptimizeVTTestParams{GetParam()};
        
        auto create_shape_constant = [](const std::vector<int64_t> & const_data, const std::string& name) {
            auto pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{const_data.size()}, const_data);
            pattern->set_friendly_name("unsqueese_pattern");
            return pattern;
        };

        // in case of non broadcast number of input channels significantly smaller
        auto numChannels = (test.kind == NetworkKind::llama3) ? 8 : 32;
        auto input_shape = test.inputShape;
        auto input_2 = static_cast<int>(test.inputShape[2]);
        auto input_3 = static_cast<int>(test.inputShape[3]);

        input_shape.at(1) = numChannels;

        // ov::Model with only a transpose node
        auto param = std::make_shared<ov::op::v0::Parameter>(test.withConvert ? ov::element::f16 : ov::element::f32, input_shape);
        param->set_friendly_name("past_key_value");

        std::shared_ptr<ov::Node> convert = test.withConvert ? 
            std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, ov::element::f32)) : 
            std::static_pointer_cast<ov::Node>(param);
        if (test.withConvert) {
            convert->set_friendly_name("convert");
        }

        // todo parametrise optional reshape
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[3] * numChannels});
        param2->set_friendly_name("new_token");

        auto reshape_pattern = create_shape_constant({0, 0, numChannels, input_3}, "reshape_pattern");
        auto transpose_pattern = create_shape_constant({1, numChannels, 1, input_3}, "transposed_pattern");

        auto reshape = std::make_shared<ov::op::v1::Reshape>(param2, test.withTranspose ? reshape_pattern : transpose_pattern, true);
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

        if (test.kind == NetworkKind::llama3) {
            auto unsqueeze_pattern =  create_shape_constant({2}, "unsqueese_pattern"); 
            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, unsqueeze_pattern);
            unsqueeze->set_friendly_name("unsqueeze");


            auto broadcast_pattern =  create_shape_constant({1, 8, 4, input_2 + 1, input_3}, "broadcast_pattern"); 
            //TODO: v1::Broadcast not working
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_pattern, ov::op::BroadcastType::BIDIRECTIONAL);
            broadcast->set_friendly_name("broadcast");

            auto reshape_pattern2 = create_shape_constant({0, 32, -1, input_3}, "reshape_pattern2");
            auto reshape2 = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_pattern2, true);
            reshape2->set_friendly_name("reshape2");

            concat_or_reshape = reshape2;
        }


        if (test.withSDPA) {
            auto mask_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[2] + 1, input_shape[2] + 1});
            mask_input->set_friendly_name("mask_input");

            auto mask_input_1 = std::make_shared<ov::op::v0::Negative>(mask_input);
            mask_input_1->set_friendly_name("mask_input_1");

            auto k_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32,  input_shape[2] + 1, input_shape[3]});
            k_input->set_friendly_name("k_input");


            auto q_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32,  input_shape[2] + 1, input_shape[3]});
            q_input->set_friendly_name("q_input");

            auto scale_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});

            // TODO: add sdpa subgraph
            std::shared_ptr<ov::Node> sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
                q_input,
                k_input,
                concat_or_reshape,
                mask_input_1,
                scale_node,
                false);
            sdpa->set_friendly_name("sdpa");
            auto result = std::make_shared<ov::op::v0::Result>(sdpa);

            result->set_friendly_name("res");
            return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, param2, mask_input, k_input, q_input});

        } else {
            auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 1, input_shape[2] + 1});
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

const std::vector<NetworkKind> networkKind = {
    // llama2 or llama3 type of concat, with convert layer or without
    NetworkKind::llama2,  NetworkKind::llama3
};

 INSTANTIATE_TEST_SUITE_P(smoke_Run_MatchAndTransposeVT,
                          TransposeVTTest,
                          ::testing::Combine(
                                ::testing::ValuesIn(input_shapes), 
                                ::testing::ValuesIn(withTranspose), 
                                ::testing::ValuesIn(withBroadCast),
                                ::testing::ValuesIn(withSDPA),
                                ::testing::ValuesIn(withHpAttenMask),
                                ::testing::ValuesIn(networkKind)),
                          TransposeVTTest::getTestCaseName);

}  // namespace
