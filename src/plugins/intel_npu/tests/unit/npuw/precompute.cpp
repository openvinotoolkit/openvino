// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include <common_test_utils/ov_test_utils.hpp>

#include "openvino/op/ops.hpp"
#include "intel_npu/config/npuw.hpp"
#include "llm_compiled_model_utils.hpp"
#include "partitioning/patterns/pre_compute.hpp"
#include "intel_npu/config/config.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

// /*
// * conditional compilation that can be used during test regression debug
// * #define ANALYZE_TEST
// * which in turn will dump subgraphs after partitioning
// */
#define ANALYZE_TEST


#ifdef ANALYZE_TEST
#define SAVE_MODEL(model, file_name) ov::save_model(model, file_name)
#else
#define SAVE_MODEL(model, file_name) 
#endif

using namespace ov::gen_pattern;
using namespace ov::pass;

namespace npuw_utest{
    using NodePtr = std::shared_ptr<ov::Node>;
}

// TODO: some patterns to be extracted from llama
enum class NetworkKind {
    prefill,
    generate
};

typedef std::tuple <
    ov::Shape,
    bool,       // hasSin
    bool,       // hasCos
    //bool,       // withSDPA - should SDPA layer present or be already unrolled or simplified
    //bool,       // use high precision on attention_mask input
    NetworkKind
> PrecomputeTestParamsTuple;

struct PrecomputeTestParams {
    #define _AT(idx) std::tuple_element<idx, PrecomputeTestParamsTuple>::type 

    _AT(0)  inputShape;
    _AT(1)  hasSin;
    _AT(2)  hasCos;
    _AT(3)  kind;
    #undef _AT

    PrecomputeTestParams(const PrecomputeTestParamsTuple& tup) {
       std::tie(inputShape, hasSin, hasCos, kind) = tup;
    }
};

static ov::OutputVector makeCosSinCache(size_t max_position_embeddings, size_t rotary_ndims) {
    std::vector<float> lut_sin(max_position_embeddings * rotary_ndims, 0.0f);
    std::vector<float> lut_cos(max_position_embeddings * rotary_ndims, 0.0f);

    // rotate_half style cos/sin table:
    //   y1 = cos(m*xita_i) * x1 - sin(m*xita_i) * x2
    //   y2 = cos(m*xita_i) * x2 + sin(m*xita_i) * x1
    //
    for (size_t i = 0, k = 0; i < rotary_ndims; i += 2, k++) {
        auto xita_i = 1.0 / std::pow(10000.0, static_cast<double>(i) / rotary_ndims);
        float* psin = lut_sin.data();
        float* pcos = lut_cos.data();
        for (size_t m = 0; m < max_position_embeddings; m++, psin += rotary_ndims, pcos += rotary_ndims) {
            auto vsin = static_cast<float>(std::sin(xita_i * m));
            auto vcos = static_cast<float>(std::cos(xita_i * m));
            pcos[k] = pcos[k + rotary_ndims / 2] = vcos;
            psin[k] = psin[k + rotary_ndims / 2] = vsin;
        }
    }
    auto Cos = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_cos);
    auto Sin = makeConst(ov::element::f32, ov::Shape({1, 1, max_position_embeddings, rotary_ndims}), lut_sin);

    return {Cos, Sin};
}

static std::shared_ptr<ov::Model> buildROPE_Llama2(const size_t batch,
                                                   const size_t seq_length,
                                                   const size_t max_position_embeddings,
                                                   const size_t ndims,
                                                   bool sin_cos_preprocessing) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{batch, seq_length, 32, ndims});
    auto param_cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});
    auto param_sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, seq_length, ndims});

    auto seq_len = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1});
    auto gather_id = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::Shape{1, seq_length});

    auto gather_from_sin_cos = [&](const ov::Output<ov::Node>& const_tab) {
        auto ScatterUpdate_152236 = makeOP<ov::opset3::ScatterUpdate>({{0, 0, 0}, {2}, seq_len, {0}});
        auto slice_Slice = makeOP<ov::opset1::StridedSlice>({const_tab, {0, 0, 0}, ScatterUpdate_152236, {1, 1, 1}},
                                                            {{"begin_mask", {1, 1, 0}},
                                                             {"end_mask", {1, 1, 0}},
                                                             {"new_axis_mask", {}},
                                                             {"shrink_axis_mask", {}},
                                                             {"ellipsis_mask", {}}});
        auto squeeze_Squeeze_435 =
            makeOP<ov::opset1::Reshape>({slice_Slice, {-1, static_cast<int>(ndims)}}, {{"special_zero", false}});
        auto index_441_Gather = makeOP<ov::opset8::Gather>({squeeze_Squeeze_435, gather_id, {0}}, {{"batch_dims", 0}});
        return makeOP<ov::opset1::Reshape>({index_441_Gather, {1, 1, -1, static_cast<int>(ndims)}},
                                           {{"special_zero", false}});
    };

    ov::OutputVector cos_sin(2);
    ov::ParameterVector parameters;
    if (sin_cos_preprocessing) {
        auto cos_sin_cache = makeCosSinCache(max_position_embeddings, ndims);
        cos_sin[0] = gather_from_sin_cos(cos_sin_cache[0]);
        cos_sin[1] = gather_from_sin_cos(cos_sin_cache[1]);
        parameters = ov::ParameterVector{input, seq_len, gather_id};
    } else {
        cos_sin[0] = param_cos;
        cos_sin[1] = param_sin;
        parameters = ov::ParameterVector{input, param_cos, param_sin};
    }

    auto transpose_Transpose = makeOP<ov::opset1::Transpose>({input, {0, 2, 1, 3}});
    auto mul_Multiply = makeOP<ov::opset1::Multiply>({transpose_Transpose, cos_sin[0]}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice_459 =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 64}, {0, 0, 0, INT_MAX}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto Constant_182988 = makeConst(ov::element::f32,
                                     ov::Shape({
                                         1,
                                         1,
                                         1,
                                         1,
                                     }),
                                     {-1.000000f});
    auto neg_Multiply = makeOP<ov::opset1::Multiply>({slice_Slice_459, Constant_182988}, {{"auto_broadcast", "numpy"}});
    auto slice_Slice =
        makeOP<ov::opset1::StridedSlice>({transpose_Transpose, {0, 0, 0, 0}, {0, 0, 0, 64}, {1, 1, 1, 1}},
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto cat_Concat = makeOP<ov::opset1::Concat>({neg_Multiply, slice_Slice}, {{"axis", -1}});
    auto mul_Multiply_463 = makeOP<ov::opset1::Multiply>({cat_Concat, cos_sin[1]}, {{"auto_broadcast", "numpy"}});
    auto add_Add = makeOP<ov::opset1::Add>({mul_Multiply, mul_Multiply_463}, {{"auto_broadcast", "numpy"}});

    return std::make_shared<ov::Model>(ov::OutputVector{add_Add}, parameters);
}

// means no sin / cos operators remained
namespace opp = ov::pass::pattern;
bool verify_rope_cached(std::shared_ptr<ov::Model> model) {
    GraphRewrite rwt;
    auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({opp::any_input()});

    auto rpe_present = false;
    auto callback = [&rpe_present](ov::pass::pattern::Matcher& m) {
        rpe_present = true;
        return false;  // root hasn't changed
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin_cos, "sin_cos_matcher");
    rwt.add_matcher<ov::pass::MatcherPass>(m, std::move(callback));
    rwt.run_on_model(model);
    return !rpe_present;
}



 class PrecomputeSinCosTest : public testing::WithParamInterface<PrecomputeTestParamsTuple>
                            , public TransformationTestsF {
public:
     void Validate() {
         auto test = PrecomputeTestParams{GetParam()};
        
         const auto isValidSubgraph  = test.hasCos && test.hasCos;
         const auto rotary_ndims = 64;

    {    
        // rope frequencies
        std::vector<float> inv_freq_values(rotary_ndims, 0.0f);
        for (size_t dim = 0; dim != rotary_ndims; dim ++) {
            auto inv_freq = 1.0 / (std::pow(10000.0,  ( 2.0 * dim / rotary_ndims)));
            inv_freq_values[dim] = inv_freq;
        }
        const auto inv_freq_const = std::make_shared<ov::opset10::Constant>(ov::element::f32, ov::Shape{1, rotary_ndims, 1}, inv_freq_values);

        const int batch = 2;
        const int seq_length = 16;
        const size_t max_position_embeddings = 2048;
        const size_t ndims = 128;
        const size_t num_head = 32;

       // model = buildROPE_Llama2(batch, seq_length, max_position_embeddings, ndims, false); 
       // SAVE_MODEL(model, "rope_with_cosin_cache.xml");
        
        ov::Core core;
        auto llama2_location_gen = "C:\\Users\\esmirno1\\Downloads\\work\\models\\Model0_00_FCEW000.xml";
        auto llama2_location_prefill = "C:\\Users\\esmirno1\\Downloads\\work\\models\\Model0_prefill_00_FCEW000.xml";

        if (test.kind == NetworkKind::prefill) {
            model = core.read_model(llama2_location_prefill);
        } else {
            model = core.read_model(llama2_location_gen);
        }

       // SAVE_MODEL(model, "model_with_rope_subgraph.xml");

    }

    {
        //TODO: build ref_model
        model_ref = model->clone();
    }
    //     auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    //     auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    //     auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    //     auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({opp::wrap_type<ov::op::v0::Constant>(), concat_1});
    //     auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::wrap_type<ov::op::v0::Constant>()});
    //     auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    //     auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    //     auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    //     auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    //     auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({concat_2});
        
    //     // zero points
    //     const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64}, 1);
    //     const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

    //     const auto right_branch = make_branch(zp_convert, {128,64,2,2}, {1, 64, 1, 1});
    //     const auto result_right = std::make_shared<opset10::Result>(right_branch);

    //     const auto left_branch = make_branch(zp_convert, {64,3,3,3}, {64, 1, 1, 1});
    //     const auto result_left = std::make_shared<opset10::Result>(left_branch);

    //     model = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    // }


    // {
    //     // zero points
    //     const auto zp_const = std::make_shared<opset10::Constant>(element::i8, ov::Shape{64}, 1);
    //     enable_keep_const_precision(zp_const);
    //     const auto zp_convert = std::make_shared<opset10::Convert>(zp_const, element::f32);

    //     const auto right_branch = make_branch(zp_convert, {128, 64, 2, 2}, {1, 64, 1, 1}, true);
    //     const auto result_right = std::make_shared<opset10::Result>(right_branch);

    //     const auto left_branch = make_branch(zp_convert, {64, 3, 3, 3}, {64, 1, 1, 1}, true);
    //     const auto result_left = std::make_shared<opset10::Result>(left_branch);

    //     model_ref = std::make_shared<Model>(ResultVector{result_right, result_left}, ParameterVector{});
    // }




        ov::npuw::patterns::pre_compute::RopeCache rpe_fuse_only{1024};
        rpe_fuse_only.run_on_model(model_ref);

        SAVE_MODEL(model_ref, "C:\\Users\\esmirno1\\Downloads\\work\\models\\model_with_rope_cached.xml");
        ASSERT_TRUE(verify_rope_cached(model_ref));  
        

        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
     }

//         ov::pass::GraphRewrite rewr;
//         rewr.add_matcher<ov::npuw::patterns::pre_compute::SinCos>();
//         rewr.run_on_model(model);

//         //ASSERT_EQ(isValidSubgraph, ov::npuw::util::optimize_value_tensors(model, test.withHpAttenMask));

//         //std::shared_ptr<ov::Model> model = ...;  // your model

//         auto test_case_name = getTestCaseName(testing::TestParamInfo<PrecomputeTestParamsTuple>{GetParam(), 0});
//         std::string xml_path = test_case_name + ".xml";
//         std::string bin_path = test_case_name + ".bin";

//         // Save the model
//         ov::pass::Serialize serialize_pass(xml_path, bin_path);
//         serialize_pass.run_on_model(model);

     
//         // validation of High Precision attention mask - implies enabling SDPA layer to be unrolled, 
//         // and also specific FP16 activation transformation in partitioning 
//         if (test.withSDPA) {
//             std::shared_ptr<::intel_npu::OptionsDesc> options_desc;

//             auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
//             auto cfg = ::intel_npu::Config(opt_desc);
//             ::intel_npu::registerNPUWOptions(*opt_desc);
//             std::map<std::string, std::string> cfg_map = {{"NPUW_F16IC", "YES"}};//, {"NPUW_ONLINE_PIPELINE", "NONE"}};
//             cfg.update(cfg_map);

//             ov::npuw::Partitioning partitioning;
//             ASSERT_NO_THROW(partitioning = ov::npuw::getPartitioning(model, cfg));

//             // input to add is 32b via convert / or via 32b parameter
//             bool bAttentionMaskVerified = false;
//             bool bAttentionMaskResultVerified = false;
            
//             auto get_rank = [](const ov::Shape& sh) {
//                 return std::count_if(sh.begin(), sh.end(), [](size_t dim) {return dim != 1;});
//             };

//             for (auto & subgraph :  partitioning.subgraphs) {
//                 auto partitioned_model = std::make_shared<ov::Model>(subgraph._results,
//                     subgraph._sinks,
//                     subgraph._parameters,
//                     "m1");
// #ifdef ANALYZE_TEST
//                 auto test_case_name = getTestCaseName(testing::TestParamInfo<OptimizeVTTestParamsTuple>{GetParam(), 0});
//                 std::string xml_path = test_case_name + "_"+ std::to_string(idx) +"_partitioned.xml";
//                 std::string bin_path = test_case_name + "_"+ std::to_string(idx) +"_partitioned.bin";
        
//                 // Save the model
//                 ov::pass::Serialize serialize_pass(xml_path, bin_path);
//                 serialize_pass.run_on_model(partitioned_model);
// #endif

//                 for (auto op : partitioned_model->get_ordered_ops()) {
//                     // case when only 1 add and 1 negate layer in whole subgraph
//                     if (ov::is_type<ov::op::v1::Add>(op)) {
//                         ASSERT_FALSE(bAttentionMaskVerified);
//                         // check rt_info
//                         // should not be any convert operation for this add
//                         // assume in lhs we have a mask
//                         auto lhs = op->get_input_node_ptr(0);
//                         auto rhs = op->get_input_node_ptr(1);
                        
//                         ASSERT_EQ(lhs->get_output_size(), 1);
//                         ASSERT_EQ(rhs->get_output_size(), 1);

//                         ASSERT_NE(lhs, nullptr) << "Add layer " << op->get_friendly_name() << " need to have two inputs";
//                         ASSERT_NE(rhs, nullptr) << "Add layer " << op->get_friendly_name() << " need to have two inputs";
                        
//                         if (get_rank(lhs->get_output_shape(0)) != 2) {
//                             ASSERT_EQ(get_rank(rhs->get_output_shape(0)), 2) 
//                                 << "Add layer " << op->get_friendly_name() << " should have 2D input, but was{" 
//                                 << lhs->get_output_shape(0) << " , " << rhs->get_output_shape(0) << " }";
//                             std::swap(lhs, rhs);
//                         }

//                         if (test.withHpAttenMask) {
//                             static constexpr char err[] = "in case of high precision, attention_mask has to accept input in fp32 without convert layer";
//                             ASSERT_TRUE(ov::is_type<ov::op::v0::Parameter>(lhs)) << err << ", actual type: " << lhs->get_type_name();
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Parameter>(lhs)->get_element_type(), ov::element::f32) << err;
//                         } else {
//                             static constexpr char err[] = "in case of fp16 precision, attention_mask should have preceding convert layer from fp16 to fp32";
//                             // input has convert from fp16 to fp32
//                             ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(lhs)) << err << ", actual type: " << lhs->get_type_name();;
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_destination_type(), ov::element::f32) << err;
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(lhs)->get_input_element_type(0), ov::element::f16) << err;
//                         }

//                         // should be only one add as atention_mask   
//                         bAttentionMaskVerified = true;
//                     }

//                     if (ov::is_type<ov::op::v0::Negative>(op)) {
//                         ASSERT_FALSE(bAttentionMaskResultVerified);
//                         // check rt_info
//                         // should not be any convert operation after this negate
//                         ASSERT_EQ(op->outputs().size(), 1);
//                         auto result_s = op->output(0).get_target_inputs();

//                         ASSERT_EQ(result_s.size(), 1);
//                         auto result = result_s.begin()->get_node();

//                         if (test.withHpAttenMask) {
//                             static constexpr char err[] = "in case of high precision, attention_mask producer need to be in fp32";
//                             ASSERT_TRUE(ov::is_type<ov::op::v0::Result>(result)) << err << ", expected type Result, actual type: " << result->get_type_name();
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Result>(result)->get_element_type(), ov::element::f32) << err;
//                         } else {
//                             static constexpr char err[] = "in case of fp16 precision, attention_mask producer should have convert layer from fp32 to fp16";
//                             ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(result)) << err << ", actual type: " << result->get_type_name();;
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_destination_type(), ov::element::f16) << err;
//                             ASSERT_EQ(ov::as_type<ov::op::v0::Convert>(result)->get_input_element_type(0), ov::element::f32) << err;
//                         }

//                         // should be only one add as atention_mask   
//                         bAttentionMaskResultVerified = true;
//                     }                    
//                 }
//             }

//             ASSERT_TRUE(bAttentionMaskVerified) << "no attention mask node not detected after applying optimize_value_tensors + run getPartitioning";
//             ASSERT_TRUE(bAttentionMaskResultVerified) << "no attention mask producer node detected after applying optimize_value_tensors + run getPartitioning";
//         }
//     }

    static std::string getTestCaseName(testing::TestParamInfo<PrecomputeTestParamsTuple> obj) {
        auto test = PrecomputeTestParams{obj.param};

        std::ostringstream result;
        result << "npuw_llm_pipeline_precompute_" << test.inputShape << "_" 
               << (test.kind == NetworkKind::generate ?  "GEN" : "PREFIL") 
               << (test.hasSin ? "_with_sin" : "")
               << (test.hasCos ? "_with_cos" : "");
        return result.str();
    }    

// protected:

//     void SetUp() override {
//         model = CreateModel();
//     }

//     std::shared_ptr<ov::Model> CreateModel() {

//         const auto test = OptimizeVTTestParams{GetParam()};
        
//         auto create_shape_constant = [](const std::vector<int64_t> & const_data, const std::string& name) {
//             auto pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{const_data.size()}, const_data);
//             pattern->set_friendly_name("unsqueese_pattern");
//             return pattern;
//         };

//         // in case of non broadcast number of input channels significantly smaller
//         auto numChannels = (test.kind == NetworkKind::llama3) ? 8 : 32;
//         auto input_shape = test.inputShape;
//         auto input_2 = static_cast<int>(test.inputShape[2]);
//         auto input_3 = static_cast<int>(test.inputShape[3]);

//         input_shape.at(1) = numChannels;

//         // ov::Model with only a transpose node
//         auto param = std::make_shared<ov::op::v0::Parameter>(test.withConvert ? ov::element::f16 : ov::element::f32, input_shape);
//         param->set_friendly_name("past_key_value");

//         std::shared_ptr<ov::Node> convert = test.withConvert ? 
//             std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, ov::element::f32)) : 
//             std::static_pointer_cast<ov::Node>(param);
//         if (test.withConvert) {
//             convert->set_friendly_name("convert");
//         }

//         // todo parametrise optional reshape
//         auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[3] * numChannels});
//         param2->set_friendly_name("new_token");

//         auto reshape_pattern = create_shape_constant({0, 0, numChannels, input_3}, "reshape_pattern");
//         auto transpose_pattern = create_shape_constant({1, numChannels, 1, input_3}, "transposed_pattern");

//         auto reshape = std::make_shared<ov::op::v1::Reshape>(param2, test.withTranspose ? reshape_pattern : transpose_pattern, true);
//         reshape->set_friendly_name("reshape");

//         std::shared_ptr<ov::Node> transpose_or_reshape;

//         if (test.withTranspose) {
//             auto constOrder = create_shape_constant({0, 2, 1, 3}, "const_order");
//             auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, constOrder);
//             transpose->set_friendly_name("transpose");
//             transpose_or_reshape = transpose;
//         } else {
//             transpose_or_reshape = reshape;
//         }

//         auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convert, transpose_or_reshape}, -2);
//         concat->set_friendly_name("concat");

//         std::shared_ptr<ov::Node> concat_or_reshape = concat;

//         if (test.kind == NetworkKind::llama3) {
//             auto unsqueeze_pattern =  create_shape_constant({2}, "unsqueese_pattern"); 
//             auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, unsqueeze_pattern);
//             unsqueeze->set_friendly_name("unsqueeze");


//             auto broadcast_pattern =  create_shape_constant({1, 8, 4, input_2 + 1, input_3}, "broadcast_pattern"); 
//             //TODO: v1::Broadcast not working
//             auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_pattern, ov::op::BroadcastType::BIDIRECTIONAL);
//             broadcast->set_friendly_name("broadcast");

//             auto reshape_pattern2 = create_shape_constant({0, 32, -1, input_3}, "reshape_pattern2");
//             auto reshape2 = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_pattern2, true);
//             reshape2->set_friendly_name("reshape2");

//             concat_or_reshape = reshape2;
//         }


//         if (test.withSDPA) {
//             auto mask_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[2] + 1, input_shape[2] + 1});
//             mask_input->set_friendly_name("mask_input");

//             auto mask_input_1 = std::make_shared<ov::op::v0::Negative>(mask_input);
//             mask_input_1->set_friendly_name("mask_input_1");

//             auto k_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32,  input_shape[2] + 1, input_shape[3]});
//             k_input->set_friendly_name("k_input");


//             auto q_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32,  input_shape[2] + 1, input_shape[3]});
//             q_input->set_friendly_name("q_input");

//             auto scale_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1});

//             // TODO: add sdpa subgraph
//             std::shared_ptr<ov::Node> sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
//                 q_input,
//                 k_input,
//                 concat_or_reshape,
//                 mask_input_1,
//                 scale_node,
//                 false);
//             sdpa->set_friendly_name("sdpa");
//             auto result = std::make_shared<ov::op::v0::Result>(sdpa);

//             result->set_friendly_name("res");
//             return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, param2, mask_input, k_input, q_input});

//         } else {
//             auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 1, input_shape[2] + 1});
//             param3->set_friendly_name("param3");

//             // TODO: what if v1 softmax???
//             auto softmax = std::make_shared<ov::op::v8::Softmax>(param3, -2);
//             softmax->set_friendly_name("softmax");

//             // entry point matmul for matcher
//             auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, concat_or_reshape);
//             matmul->set_friendly_name("matmul");

//             auto result = std::make_shared<ov::op::v0::Result>(matmul);
//             result->set_friendly_name("res");
//             return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, param2, param3});
    
//         }
//     }    

//     std::shared_ptr<ov::Model> model;
};

class PrecomputeValidateInverseFreq : public testing::WithParamInterface<PrecomputeTestParamsTuple>, public ov::test::TestsCommon {
     
public:
     void Validate() {
         auto test = PrecomputeTestParams{GetParam()};
        
         const auto isValidSubgraph  = test.hasCos && test.hasCos;
         const auto rotary_ndims = 64;

          // rope frequencies base=100 perfectly matches llama 3.2 style
        std::vector<float> inv_freq_values(rotary_ndims, 0.0f);
        for (size_t dim = 0; dim != rotary_ndims; dim ++) {
            auto inv_freq = 1.0 / (std::pow(1000.0,  ( 2.0 * dim / rotary_ndims)));
            inv_freq_values[dim] = inv_freq;
        }
        const auto inv_freq_const = std::make_shared<ov::opset10::Constant>(ov::element::f32, ov::Shape{1, rotary_ndims, 1}, inv_freq_values);

        const int batch = 2;
        const int seq_length = 16;
        const size_t max_position_embeddings = 2048;
        const size_t ndims = 128;
        const size_t num_head = 32;

       // TODO: restore build-rope function 
       // model = buildROPE_Llama2(batch, seq_length, max_position_embeddings, ndims, false); 
       // SAVE_MODEL(model, "rope_with_cosin_cache.xml");
        
        ov::Core core;
        // fixed shapes llama 3.2 head
        auto llama2_location = "C:\\Users\\esmirno1\\Downloads\\work\\models\\Model0_00_FCEW000.xml";
        // model = core.read_model("qwen1-7b/openvino_model.xml");
        auto model = core.read_model(llama2_location);

        using CPtr = std::shared_ptr<ov::op::v0::Constant>;
        std::vector<CPtr> to_keep;
    
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::pre_compute::RopeInverseFreq>(std::ref(to_keep));
        rewr.run_on_model(model);

        ASSERT_TRUE(!to_keep.empty());

        auto values = to_keep.front()->cast_vector<float>();
        ASSERT_EQ(values.size(), inv_freq_values.size());

        for (int i = 0; i < values.size(); ++i) {
            //std::cout<< "index: " << i << " v= " << values[i] << ", ref="<< inv_freq_values[i] << "\n";
            ASSERT_NEAR(values[i], inv_freq_values[i], 1e-5) << "Mismatch at index: " << i << ", by: " << std::abs(values[i] - inv_freq_values[i]);
        }
     }
};


TEST_P(PrecomputeSinCosTest, smoke_Run_MatchAndPrecomputeSinCos) {
     Validate();
}
TEST_P(PrecomputeValidateInverseFreq, smoke_Run_MatchInverseFreqInRope) {
    Validate();
}

 
namespace {
// eliminate direct shape dependency to match llama2, as in test and in optimize function
const std::vector<ov::Shape> input_shapes{{1, 0, 1024, 128}, {1, 0, 1141, 64}};

const std::vector<bool> hasSin{true, false};
const std::vector<bool> hasCos{true, false};


const std::vector<NetworkKind> networkKind = {
    NetworkKind::prefill,  
    NetworkKind::generate
};

 INSTANTIATE_TEST_SUITE_P(smoke_Run_MatchAndPrecomputeSinCos,
                          PrecomputeSinCosTest,
                          ::testing::Combine(
                                ::testing::ValuesIn(input_shapes), 
                                ::testing::ValuesIn(hasSin), 
                                ::testing::ValuesIn(hasCos),
                                ::testing::ValuesIn(networkKind)),
                          PrecomputeSinCosTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Run_PrecomputeValidateInverseFreq,
    PrecomputeValidateInverseFreq,
    ::testing::Combine(
            ::testing::ValuesIn(input_shapes), 
            ::testing::ValuesIn(hasSin), 
            ::testing::ValuesIn(hasCos),
            ::testing::ValuesIn(networkKind)));

}  // namespace





