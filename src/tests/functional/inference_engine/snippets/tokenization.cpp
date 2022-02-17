// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include <snippets/snippets_isa.hpp>
#include <snippets/pass/collapse_subgraph.hpp>
#include <snippets/op/subgraph.hpp>

#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace testing;
using namespace ov;
using ngraph::snippets::op::Subgraph;
using ngraph::pass::InitNodeInfo;
using ngraph::snippets::pass::EnumerateNodes;
using ngraph::snippets::pass::TokenizeSnippets;
//using ngraph::snippets::pass::CreateSubgraph;

// Todo: Move this test to CPU-specific
TEST(TransformationTests, DoNotStartSubgraphAfterInputs) {
    // Do not start Subgraph after input parameters to avoid U8->FP32 and FP32->U8 conversion pairs
    // Todo: Remove this test when U8 support is enabled in SnippetS and StartSubgraph logics is updated
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        const std::vector<float> const_values{3, 2, 10};
        auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
        auto add = std::make_shared<op::v1::Add>(data0, data1);
        auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
        auto mul = std::make_shared<op::v1::Multiply>(add, sub);
        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        // Todo: When moved to CPU-specific tests, uncomment the markup transformation below.
        //  m.register_pass<MKLDNNPlugin::SnippetsMarkFused>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    ASSERT_EQ(count_ops_of_type<Subgraph>(f), 0);
}

TEST(TransformationTests, StartSubgraphMultipleOutputs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 3});
        auto convert0 = std::make_shared<op::v0::Convert>(data0, element::f32);
        auto convert1 = std::make_shared<op::v0::Convert>(data1, element::f32);
        const std::vector<float> const_values{3, 2, 10};
        auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
        auto add = std::make_shared<op::v1::Add>(convert0, convert1);
        auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
        auto mul = std::make_shared<op::v1::Multiply>(add, sub);
        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 3});
        auto convert0 = std::make_shared<op::v0::Convert>(data0, element::f32);
        auto convert1 = std::make_shared<op::v0::Convert>(data1, element::f32);
        const std::vector<float> const_values{3, 2, 10};
        auto const_data = std::make_shared<op::v0::Constant>(element::f32, Shape{1, 3}, const_values);
        auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto indata2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<op::v1::Add>(indata0, indata1);
        auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
        auto mul = std::make_shared<Subgraph>(NodeVector{convert0, convert1, const_data},
       std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Multiply>(add, sub)},
                                  ParameterVector{indata0, indata1, indata2}));
        f_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, TokenizeMulAddSubgraph) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
        auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
        auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);

        auto mul_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
        auto mul_1 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_1);
        auto add_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
        auto add_1 = std::make_shared<op::v1::Add>(mul_1, add_const_1);
        auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

        auto mul_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
        auto mul_2 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_2);
        auto add_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
        auto add_2 = std::make_shared<op::v1::Add>(mul_2, add_const_2);
        auto relu = std::make_shared<op::v0::Relu>(add_2);

        auto add = std::make_shared<op::v1::Add>(elu, relu);
        auto result = std::make_shared<op::v0::Result>(add);

        f = std::make_shared<Model>(ResultVector{ result }, ParameterVector{ data_1, data_2 });

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });
        auto data_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });

        // snippet inputs
        auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);
        auto mul_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
        auto add_const_1 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });
        auto mul_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 4.f });
        auto add_const_2 = op::v0::Constant::create(element::f32, { 1, 3, 1, 1 }, { 16.4f });

        // snippet function
        auto snippet_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 4, 4 });

        auto sn_mul_const_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
        auto mul_1 = std::make_shared<op::v1::Multiply>(snippet_input, sn_mul_const_1);
        auto sn_add_const_1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
        auto add_1 = std::make_shared<op::v1::Add>(mul_1, sn_add_const_1);
        auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

        auto sn_mul_const_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
        auto mul_2 = std::make_shared<op::v1::Multiply>(snippet_input, sn_mul_const_2);
        auto sn_add_const_2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{ 1, 3, 1, 1 });
        auto add_2 = std::make_shared<op::v1::Add>(mul_2, sn_add_const_2);
        auto relu = std::make_shared<op::v0::Relu>(add_2);

        auto add = std::make_shared<op::v1::Add>(elu, relu);
        ParameterVector subgraph_params{ snippet_input, sn_mul_const_1, sn_add_const_1, sn_mul_const_2, sn_add_const_2 };
        auto snippet_function = std::make_shared<Model>(NodeVector{ add }, subgraph_params);

        ngraph::NodeVector snippet_inputs{ non_snippet_op, mul_const_1, add_const_1, mul_const_2, add_const_2 };
        auto snippet = std::make_shared<Subgraph>(snippet_inputs, snippet_function);
        auto result = std::make_shared<op::v0::Result>(snippet);

        f_ref = std::make_shared<Model>(NodeVector{ result }, ParameterVector{ data_1, data_2 });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, DontStartSubgraphSingleOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<op::v1::Add>(data0, data1);
        auto sub = std::make_shared<op::v1::Subtract>(add, data1);
        auto mul = std::make_shared<op::v1::Multiply>(data0, sub);
        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<op::v1::Add>(data0, data1);
        auto sub = std::make_shared<op::v1::Subtract>(add, data1);
        auto mul = std::make_shared<op::v1::Multiply>(data0, sub);
        f_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, AttachToSubgraph) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<Subgraph>(NodeVector{data0, data1},
            std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto neg = std::make_shared<op::v0::Negative>(add);
        auto concat = std::make_shared<op::v0::Concat>(NodeVector{add, neg}, 0);
        f = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto inner = std::make_shared<op::v1::Add>(indata0, indata1);
        auto add = std::make_shared<Subgraph>(NodeVector{data0, data1},
            std::make_shared<Model>(NodeVector{std::make_shared<op::v0::Negative>(inner), inner}, ParameterVector{indata0, indata1}));
        auto concat = std::make_shared<op::v0::Concat>(OutputVector{add->output(0), add->output(1)}, 0);
        f_ref = std::make_shared<Model>(NodeVector{concat}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, DontAttachToSubgraphIfLoop) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<Subgraph>(NodeVector{data0, data1},
            std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto log = std::make_shared<op::v0::Log>(add);
        auto mul = std::make_shared<op::v1::Multiply>(add, log);
        f = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});

        pass::Manager m;
        m.register_pass<InitNodeInfo>();
        m.register_pass<EnumerateNodes>();
        m.register_pass<TokenizeSnippets>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto data1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto indata0 = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
        auto indata1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
        auto add = std::make_shared<Subgraph>(NodeVector{data0, data1},
            std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Add>(indata0, indata1)}, ParameterVector{indata0, indata1}));
        auto log = std::make_shared<op::v0::Log>(add);
        /*
         * Note that log is not currently supported by snippets, so it won't be converted to subgraph.
         * Mul will be converted for the "reset" continuation strategy, (present case)
         * or left as-is for the "abort" continuation strategy
        */
        auto add_param = std::make_shared<op::v0::Parameter>(element::f32, add->get_output_shape(0));
        auto log_param = std::make_shared<op::v0::Parameter>(element::f32, log->get_output_shape(0));
        auto mul = std::make_shared<Subgraph>(NodeVector{add, log},
                   std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Multiply>(add_param, log_param)},
                                                ParameterVector{add_param, log_param}));
        f_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
