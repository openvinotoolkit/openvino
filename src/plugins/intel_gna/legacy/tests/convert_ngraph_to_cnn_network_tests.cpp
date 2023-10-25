// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp/ie_cnn_network.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <legacy/cnn_network_impl.hpp>  // deprecated API
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>

using namespace testing;
using namespace InferenceEngine;

TEST(ConvertFunctionToCNNNetworkTests, ConvertPReLUNetwork) {
    std::shared_ptr<ov::Model> f;
    {
        auto param1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 2});
        auto param2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 2});
        auto prelu = std::make_shared<ov::opset1::PRelu>(param1, param2);
        prelu->set_friendly_name("prelu");
        auto result = std::make_shared<ngraph::op::Result>(prelu);

        f = std::make_shared<ov::Model>(ngraph::ResultVector{result}, ov::ParameterVector{param1, param2});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.run_passes(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(nGraphImpl);
    } catch (InferenceEngine::Exception& err) {
        const std::string ref_msg =
            "Error of validate layer: prelu with type: PReLU. Number of inputs (2) is not equal to expected ones: 1";
        const std::string resp_msg = err.what();
        ASSERT_TRUE(resp_msg.find(ref_msg) != std::string::npos) << resp_msg;
    }
}

TEST(ConvertFunctionToCNNNetworkTests, ConvertConvolutionNetwork) {
    std::shared_ptr<ov::Model> f;
    {
        auto param1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});
        auto param2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 3, 1, 1});
        auto convolution = std::make_shared<ngraph::op::ConvolutionIE>(param1,
                                                                       param2,
                                                                       ov::Strides{1, 1},
                                                                       ov::Strides{1, 1},
                                                                       ov::CoordinateDiff{0, 0},
                                                                       ov::CoordinateDiff{0, 0},
                                                                       ov::element::f32);
        convolution->set_friendly_name("convolution");
        auto result = std::make_shared<ngraph::op::Result>(convolution);

        f = std::make_shared<ov::Model>(ngraph::ResultVector{result}, ov::ParameterVector{param1, param2});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.run_passes(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(nGraphImpl);
    } catch (InferenceEngine::Exception& err) {
        FAIL() << err.what();
    }
}

TEST(ConvertFunctionToCNNNetworkTests, OpsShouldBeConvertedToIERepresentation) {
    ov::NodeVector should_converted_to_ie = {
        std::make_shared<ov::opset4::Convolution>(),
        std::make_shared<ov::opset4::GatherTree>(),
        std::make_shared<ov::opset4::GroupConvolution>(),
        std::make_shared<ov::opset4::GroupConvolutionBackpropData>(),
        std::make_shared<ov::opset4::GRUCell>(),
        // std::make_shared<ngraph::op::v5::GRUSequence>(), todo: enable after GRUSequence support
        std::make_shared<ov::opset4::HardSigmoid>(),
        std::make_shared<ov::opset4::LRN>(),
        std::make_shared<ov::opset4::LSTMCell>(),
        // std::make_shared<ngraph::op::v5::LSTMSequence>(), todo: enable after LSTMSequence support
        std::make_shared<ov::opset4::NonMaxSuppression>(),
        std::make_shared<ov::opset4::NormalizeL2>(),
        std::make_shared<ov::opset4::RNNCell>(),
        // std::make_shared<ngraph::op::v5::RNNSequence>(), todo: enable after RNNSequence support
        std::make_shared<ov::opset4::OneHot>(),
        std::make_shared<ov::opset4::Pad>(),
        std::make_shared<ov::opset4::PriorBoxClustered>(),
        std::make_shared<ov::opset4::PriorBox>(),
        std::make_shared<ov::opset4::Proposal>(),
        std::make_shared<ov::opset4::Selu>(),
        std::make_shared<ov::opset4::Swish>(),
    };

    // create simple ngraph function Parameter -> Result
    std::shared_ptr<ov::Model> f;
    auto param = std::make_shared<ov::opset4::Parameter>(ov::element::f32, ov::Shape{});
    auto res = std::make_shared<ov::opset4::Result>(param);
    f = std::make_shared<ov::Model>(ngraph::ResultVector{res}, ov::ParameterVector{param});
    InferenceEngine::CNNNetwork nGraphImpl(f);

    for (const auto& ngraph_node : should_converted_to_ie) {
        // add node without inputs to the ngraph function
        ngraph_node->set_output_type(0, ov::element::f32, ov::Shape{});
        res->input(0).replace_source_output(ngraph_node->output(0));

        EXPECT_THROW(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl, true),
                     InferenceEngine::Exception)
            << "failed node: " << ngraph_node->get_type_name() << std::endl;
        try {
            InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl, true);
        } catch (InferenceEngine::Exception& err) {
            std::string type_name = ngraph_node->get_type_name();

            std::map<std::string, std::string> exceptions = {
                {"Broadcast", "Tile"},
                {"Interpolate", "Interp"},
                {"NormalizeL2", "NormalizeIE"},
                {"GroupConvolution", "ConvolutionIE"},
                {"ConvolutionBackpropData", "DeconvolutionIE"},
                {"GroupConvolutionBackpropData", "DeconvolutionIE"},
            };
            std::string type_name_ie = type_name + "IE";
            if (exceptions[type_name].empty()) {
                type_name_ie = type_name + "IE";
            } else {
                type_name_ie = exceptions[type_name];
            }
            std::string expected_error_message = type_name + " operation has a form that is not supported. " +
                                                 ngraph_node->get_friendly_name() + " should be converted to " +
                                                 type_name_ie + " operation.";
            std::string real_message = err.what();
            bool is_messages_match = real_message.find(expected_error_message) != std::string::npos;
            EXPECT_TRUE(is_messages_match) << "failed node: " << type_name << std::endl
                                           << "Exception massage: " << err.what() << std::endl
                                           << "Expected message: " << expected_error_message << std::endl;
        } catch (...) {
            FAIL() << "ERROR: Unexpected exception thrown: " << std::current_exception << std::endl;
        }
    }
}

TEST(ConvertFunctionToCNNNetworkTests, ConvertTopKWithOneInput) {
    std::shared_ptr<ov::Model> f;
    {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        ov::Shape const_shape = {};
        std::vector<int64_t> val = {5};
        auto k = std::make_shared<ov::opset4::Constant>(ov::element::i64, const_shape, val);
        auto topK = std::make_shared<ov::opset4::TopK>(param,
                                                       k,
                                                       2,
                                                       ov::opset4::TopK::Mode::MAX,
                                                       ov::opset4::TopK::SortType::SORT_VALUES);
        topK->set_friendly_name("topK");
        auto result = std::make_shared<ngraph::op::Result>(topK->output(1));

        f = std::make_shared<ov::Model>(ngraph::ResultVector{result}, ov::ParameterVector{param});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.run_passes(f);
    }

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.register_pass<ov::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ov::pass::ConvertOpSet2ToOpSet1>();

    static const precisions_map convert_precision_map{
        {ov::element::i64, ov::element::i32},
        {ov::element::u64, ov::element::i32},
        {ov::element::u16, ov::element::i32},
        {ov::element::u32, ov::element::i32},
        {ov::element::f16, ov::element::f32},
        {ov::element::boolean, ov::element::u8},
    };

    manager.register_pass<ov::pass::ConvertPrecision>(convert_precision_map);
    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    manager.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::i64, ov::element::i32}});

    manager.run_passes(f);

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    IE_SUPPRESS_DEPRECATED_END

    try {
        OutputsDataMap outputs = nGraphImpl.getOutputsInfo();
        ASSERT_EQ(outputs.size(), 1);
        ASSERT_EQ(outputs.begin()->first, "topK.1");
    } catch (InferenceEngine::Exception& err) {
        const std::string ref_msg =
            "Error of validate layer: prelu with type: PReLU. Number of inputs (2) is not equal to expected ones: 1";
        const std::string resp_msg = err.what();
        ASSERT_TRUE(resp_msg.find(ref_msg) != std::string::npos) << resp_msg;
    }
}

TEST(ConvertFunctionToCNNNetworkTests, UnsupportedDynamicOps) {
    std::shared_ptr<ov::Model> f;
    {
        auto param = std::make_shared<ov::opset4::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        param->set_friendly_name("param");
        auto relu = std::make_shared<ov::opset4::Relu>(param);
        relu->set_friendly_name("relu");
        auto non_zero = std::make_shared<ov::opset4::NonZero>(relu);
        non_zero->set_friendly_name("non_zero");
        auto result = std::make_shared<ngraph::op::Result>(non_zero->output(0));
        result->set_friendly_name("result");

        f = std::make_shared<ov::Model>(ngraph::ResultVector{result}, ov::ParameterVector{param});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl);
        FAIL() << "InferenceEngine::Exception must be thrown";
    } catch (InferenceEngine::Exception& e) {
        EXPECT_THAT(e.what(),
                    testing::HasSubstr(std::string("Unsupported dynamic ops: \n"
                                                   "opset1::Parameter param () -> (f32[...])\n"
                                                   "opset1::Relu relu (param[0]:f32[...]) -> (f32[...])\n"
                                                   "opset3::NonZero non_zero (relu[0]:f32[...]) -> (i64[?,?])\n"
                                                   "opset1::Result result (non_zero[0]:i64[?,?]) -> (i64[?,?])")));
    }
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesAllInternal) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
        auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        end->set_friendly_name(begin->get_name());
        auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
        auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                             begin,
                                                             end,
                                                             stride,
                                                             std::vector<int64_t>{1, 1},
                                                             std::vector<int64_t>{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{ss}, ov::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(nGraphImpl.layerCount(), 5);
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesHasResult1) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
        auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        end->set_friendly_name(begin->get_name());
        auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
        auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                             begin,
                                                             end,
                                                             stride,
                                                             std::vector<int64_t>{1, 1},
                                                             std::vector<int64_t>{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{ss, begin}, ov::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(nGraphImpl.layerCount(), 5);
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesHasResult2) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
        auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        begin->set_friendly_name("const");
        auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        end->set_friendly_name("const");
        auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
        stride->set_friendly_name("const");
        auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                             begin,
                                                             end,
                                                             stride,
                                                             std::vector<int64_t>{1, 1},
                                                             std::vector<int64_t>{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{ss, begin}, ov::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(nGraphImpl.layerCount(), 5);
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesHasResult3) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
        auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        begin->set_friendly_name("const");
        auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        end->set_friendly_name("const");
        auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
        stride->set_friendly_name("const");
        auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                             begin,
                                                             end,
                                                             stride,
                                                             std::vector<int64_t>{1, 1},
                                                             std::vector<int64_t>{1, 1});
        ss->set_friendly_name("node");
        auto squeeze =
            std::make_shared<ov::opset1::Squeeze>(ss, ov::opset1::Constant::create(ov::element::i64, {1}, {0}));
        squeeze->set_friendly_name("node");
        f = std::make_shared<ov::Model>(ov::NodeVector{squeeze, begin}, ov::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(nGraphImpl.layerCount(), 7);
    auto outputs_info = nGraphImpl.getOutputsInfo();
    ASSERT_TRUE(outputs_info.count("node"));
    ASSERT_TRUE(outputs_info.count("const"));
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesNegative) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
        auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        begin->set_friendly_name("const");
        auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
        end->set_friendly_name("const");
        auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
        auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                             begin,
                                                             end,
                                                             stride,
                                                             std::vector<int64_t>{1, 1},
                                                             std::vector<int64_t>{1, 1});

        f = std::make_shared<ov::Model>(ov::NodeVector{ss, begin, end}, ov::ParameterVector{input});
    }

    ASSERT_THROW(InferenceEngine::CNNNetwork{f}, InferenceEngine::Exception);
}

TEST(ConvertFunctionToCNNNetworkTests, NonUniqueNamesParametersNegative) {
    std::shared_ptr<ov::Model> f(nullptr);
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
    input->set_friendly_name("param");
    auto begin = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
    auto end = ov::opset1::Constant::create(ov::element::i64, {2}, {0, 0});
    auto stride = ov::opset1::Constant::create(ov::element::i64, {2}, {1, 1});
    auto ss = std::make_shared<ov::opset1::StridedSlice>(input,
                                                         begin,
                                                         end,
                                                         stride,
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{1, 1});
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3});
    auto concat = std::make_shared<ov::opset1::Concat>(ov::NodeVector{ss, input2}, 0);

    f = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input, input2});

    InferenceEngine::CNNNetwork nGraphImpl(f);
    try {
        input2->set_friendly_name("param");
        InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl);
        FAIL() << "InferenceEngine::Exception must be thrown";
    } catch (InferenceEngine::Exception& e) {
        EXPECT_THAT(e.what(), testing::HasSubstr(std::string("Detected two output operations with the same name:")));
    }
}

TEST(ConvertFunctionToCNNNetworkTests, IteratorForMemoryLayers) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto constReadVal = ov::opset5::Constant::create(ov::element::f32, {1, 37632}, {0});
        constReadVal->set_friendly_name("const");
        auto readVal = std::make_shared<ov::opset5::ReadValue>(constReadVal, "buffer_1");
        readVal->set_friendly_name("readVal_Buf1");

        auto constVarSplit1 = ov::opset5::Constant::create(ov::element::i64, {}, {1});
        constVarSplit1->set_friendly_name("varSplitConst1");
        auto constVarSplit2 = ov::opset5::Constant::create(ov::element::i64, {2}, {5376, 32256});
        constVarSplit2->set_friendly_name("varSplitConst2");

        auto varSplit = std::make_shared<ov::opset5::VariadicSplit>(readVal, constVarSplit1, constVarSplit2);

        auto param1 = std::make_shared<ov::opset5::Parameter>(ov::element::f32, ov::Shape{1, 5376});
        auto varConcat = std::make_shared<ov::opset5::Concat>(ov::OutputVector{varSplit->output(0), param1}, 1);
        auto result = std::make_shared<ov::opset5::Result>(varConcat);

        auto param2 = std::make_shared<ov::opset5::Parameter>(ov::element::f32, ov::Shape{1, 5376});
        auto varConcat2 = std::make_shared<ov::opset5::Concat>(ov::OutputVector{varSplit->output(1), param2}, 1);

        auto assign = std::make_shared<ov::opset5::Assign>(varConcat2, "buffer_1");
        f = std::make_shared<ov::Model>(ngraph::ResultVector{result},
                                        ngraph::SinkVector{assign},
                                        ov::ParameterVector{param1, param2});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    int memory_count(0);
    for (details::CNNNetworkIterator itLayer{nGraphImpl}; itLayer != details::CNNNetworkIterator(); itLayer++) {
        if ((*itLayer)->type == "Memory")
            memory_count++;
    }
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(2, memory_count);
}

TEST(ConvertFunctionToCNNNetworkTests, IteratorForMemoryLayers2) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto constReadVal = ov::opset5::Constant::create(ov::element::f32, {1, 37632}, {0});
        constReadVal->set_friendly_name("const");
        auto readVal = std::make_shared<ov::opset5::ReadValue>(constReadVal, "buffer_1");
        readVal->set_friendly_name("readVal_Buf1");

        auto constVarSplit1 = ov::opset5::Constant::create(ov::element::i64, {}, {1});
        constVarSplit1->set_friendly_name("varSplitConst1");
        auto constVarSplit2 = ov::opset5::Constant::create(ov::element::i64, {2}, {5376, 32256});
        constVarSplit2->set_friendly_name("varSplitConst2");

        auto varSplit = std::make_shared<ov::opset5::VariadicSplit>(readVal, constVarSplit1, constVarSplit2);

        auto param2 = std::make_shared<ov::opset5::Parameter>(ov::element::f32, ov::Shape{1, 5376});
        auto varConcat2 = std::make_shared<ov::opset5::Concat>(ov::OutputVector{varSplit->output(1), param2}, 1);

        auto assign = std::make_shared<ov::opset5::Assign>(varConcat2, "buffer_1");
        f = std::make_shared<ov::Model>(ngraph::ResultVector{},
                                        ngraph::SinkVector{assign},
                                        ov::ParameterVector{param2});
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);
    IE_SUPPRESS_DEPRECATED_START
    nGraphImpl = CNNNetwork(InferenceEngine::details::convertFunctionToICNNNetwork(f, nGraphImpl));
    int memory_count(0);
    for (details::CNNNetworkIterator itLayer{nGraphImpl}; itLayer != details::CNNNetworkIterator(); itLayer++) {
        if ((*itLayer)->type == "Memory")
            memory_count++;
    }
    IE_SUPPRESS_DEPRECATED_END
    ASSERT_EQ(2, memory_count);
}
