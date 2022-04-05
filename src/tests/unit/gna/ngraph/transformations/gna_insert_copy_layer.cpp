// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>
#include "transformations/insert_copy_layer.hpp"
#include "ops/copy.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <transformations/utils/utils.hpp>

namespace testing {

typedef std::tuple<
        size_t,    // Concat axis
        size_t     // input number
> InsertCopyTestParams;


class InsertCopyLayerTest: public CommonTestUtils::TestsCommon,
                         public ::testing::WithParamInterface<InsertCopyTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertCopyTestParams>& obj) {
        size_t axis, inputs_num;
        std::tie(axis, inputs_num) = obj.param;

        std::ostringstream result;
        result << "inputsNum=" << inputs_num << "_";
        result << "axis=" << axis;

        return result.str();
    }
    void SetUp() override;
    virtual void Validate();
    virtual void Run();
public:
    std::shared_ptr<ngraph::Function> m_func, m_ref_func;
    size_t m_axis, m_inputs_num;
};

void InsertCopyLayerTest::Validate() {
    ASSERT_NO_THROW(check_rt_info(m_func));

    auto result = compare_functions(m_func, m_ref_func);
    ASSERT_TRUE(result.first);
}

void InsertCopyLayerTest::SetUp() {
    std::tie(m_axis, m_inputs_num) = this->GetParam();
}

void InsertCopyLayerTest::Run() {
    SetUp();
    Validate();
}

class InsertCopyLayerConcatTest: public InsertCopyLayerTest {
public:
    void SetUp() override {
        ngraph::Shape input_shape{10};
        InsertCopyLayerTest::SetUp();

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            ngraph::OutputVector concat_inputs;
            for (int i = 0; i < m_inputs_num; ++i) {
                concat_inputs.push_back(add);
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);
            ngraph::OutputVector concat_inputs = {};
            for (int i = 0; i < m_inputs_num - 1; ++i) {
                concat_inputs.push_back(copy);
            }
            concat_inputs.push_back(add);
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }
    }

    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(m_func);

       InsertCopyLayerTest::Validate();
    }
};

class InsertCopyLayerSplitConcatTest: public InsertCopyLayerTest {
public:
    void SetUp() override {
        ngraph::Shape input_shape{256};
        InsertCopyLayerTest::SetUp();

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, m_inputs_num, m_axis);

            ngraph::OutputVector concat_inputs;
            for (int i = 0; i < m_inputs_num; ++i) {
                concat_inputs.push_back(split->output(i));
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
            auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, m_inputs_num, m_axis);

            ngraph::OutputVector concat_inputs;
            for (int i = 0; i < m_inputs_num; ++i) {
                if (m_inputs_num == 1 || (i % (m_inputs_num / 8) == 0))
                    concat_inputs.push_back(std::make_shared<ov::intel_gna::op::Copy>(split->output(i)));
                else
                    concat_inputs.push_back(split->output(i));
            }
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, m_axis);

            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            m_ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeConcatLayer>();
        m.run_passes(m_func);

       InsertCopyLayerTest::Validate();
    }
};

TEST(TransformationTests, InsertCopyLayerMultiParamConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        size_t axis = 0;
        ngraph::Shape in_shape{10};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            ngraph::OutputVector concat_inputs{params, params};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);

            ngraph::OutputVector concat_inputs{copy, copy};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamNFLConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        size_t axis = 0;
        ngraph::Shape shape    = {1, 1, 2, 4};
        ngraph::Shape in_shape = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(params, shape);
            ngraph::OutputVector concat_inputs{reshape1, reshape2};

            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(params, shape);
            auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
            auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

            ngraph::OutputVector concat_inputs{copy1, copy2};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamMultiNFLConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        size_t axis = 0;
        ngraph::Shape shape    = {1, 1, 2, 4};
        ngraph::Shape in_shape = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(params, shape);
            ngraph::OutputVector concat_inputs{reshape1, reshape2};

            auto concat1 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto concat2 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result1 = std::make_shared<ngraph::opset8::Result>(concat1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(concat2);
            auto result3 = std::make_shared<ngraph::opset8::Result>(reshape1);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(params, shape);
            auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
            auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

            ngraph::OutputVector concat_inputs{copy1, copy2};
            auto concat1 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto concat2 = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result1 = std::make_shared<ngraph::opset8::Result>(concat1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(concat2);
            auto result3 = std::make_shared<ngraph::opset8::Result>(reshape1);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiConstConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func1, ref_func2;
        size_t axis = 0;
        ngraph::Shape in_shape{10};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);

            ngraph::OutputVector concat_inputs{params, constant, constant};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(constant);

            ngraph::OutputVector concat_inputs{params, copy, constant};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func1 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(constant);

            ngraph::OutputVector concat_inputs{params, constant, copy};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func2 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeConcatLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result1 = compare_functions(func, ref_func1);
        auto result2 = compare_functions(func, ref_func2);
        ASSERT_TRUE(result1.first || result2.first);
}

TEST(TransformationTests, InsertCopyLayerMultiLayerConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func1, ref_func2;
        size_t axis = 0;
        ngraph::Shape in_shape{10};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            ngraph::OutputVector concat_inputs{add, add};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);

            ngraph::OutputVector concat_inputs{copy, add};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func1 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(add);

            ngraph::OutputVector concat_inputs{add, copy};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func2 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        // Transformation is based on outputs order and insert copy layer in one of the branches,
        // so this is right, that we have two different result graph based on output order.
        auto result1 = compare_functions(func, ref_func1);
        auto result2 = compare_functions(func, ref_func2);

        ASSERT_TRUE(result1.first || result2.first);
}

TEST(TransformationTests, InsertCopyLayerMultiLayerNFLConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func1, ref_func2;
        size_t axis = 0;
        ngraph::Shape shape    = {1, 1, 2, 4};
        ngraph::Shape in_shape = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto reshape1 = ngraph::op::util::reshapeTo(add, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(add, shape);
            ngraph::OutputVector concat_inputs{reshape1, reshape2};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto reshape1 = ngraph::op::util::reshapeTo(add, shape);
            auto reshape_copy = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
            auto reshape2 = ngraph::op::util::reshapeTo(add, shape);

            ngraph::OutputVector concat_inputs{reshape_copy, reshape2};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func1 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto add = std::make_shared<ngraph::opset8::Add>(params, params);
            auto reshape1 = ngraph::op::util::reshapeTo(add, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(add, shape);
            auto reshape_copy = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

            ngraph::OutputVector concat_inputs{reshape1, reshape_copy};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concat_inputs, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func2 = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        // Transformation is based on outputs order and insert copy layer in one of the branches,
        // so this is right, that we have two different result graph based on output order.
        auto result1 = compare_functions(func, ref_func1);
        auto result2 = compare_functions(func, ref_func2);

        ASSERT_TRUE(result1.first || result2.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(input, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto add = std::make_shared<ngraph::opset8::Add>(input, read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(add);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(input);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto add = std::make_shared<ngraph::opset8::Add>(input, read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(add);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamConcatMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(input, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(input);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamNFLConcatMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(input, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(input, shape);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(reshape1, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);

            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{reshape2}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(input, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(input, shape);
            auto copy1 = std::make_shared<ov::intel_gna::op::Copy>(reshape1);
            auto copy2 = std::make_shared<ov::intel_gna::op::Copy>(reshape2);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy1, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);

            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy2}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiLayerConcatMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(input, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{read_value}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(input, "variable_id");
            auto copy_rv = std::make_shared<ov::intel_gna::op::Copy>(read_value);
            auto assign = std::make_shared<ngraph::opset3::Assign>(copy_rv, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{read_value}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerMultiParamLayerConcatMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(input, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input, read_value}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(input);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto copy_rv = std::make_shared<ov::intel_gna::op::Copy>(read_value);
            auto assign = std::make_shared<ngraph::opset3::Assign>(copy_rv, "variable_id");
            assign->add_control_dependency(read_value);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy, read_value}, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleMultiConnectedLayerToConcatAndMemory>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerCropMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(input, shape);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(crop, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(input, shape);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);

            auto copy = std::make_shared<ov::intel_gna::op::Copy>(crop);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerCropNFLMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape1        = {1, 1, 2, 4};
        ngraph::Shape shape2        = {1, 1, 2, 2};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(input, shape1);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
            auto reshape2 = ngraph::op::util::reshapeTo(crop, shape2);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(reshape2, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(input, shape1);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
            auto reshape2 = ngraph::op::util::reshapeTo(crop, shape2);

            auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape2);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerConcatMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input}, axis);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(concat, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input}, axis);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(concat);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerConcatNFLMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape shape    = {1, 1, 2, 4};
        ngraph::Shape in_shape = {1, 2, 4};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input}, axis);
            auto reshape = ngraph::op::util::reshapeTo(concat, shape);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(reshape, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input}, axis);
            auto reshape = ngraph::op::util::reshapeTo(concat, shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerSplitMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape in_shape{10};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(split, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(split);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(split);
            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(split);


            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerSplitNFLMemoryTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        ngraph::Shape shape    = {1, 1, 2, 4};
        ngraph::Shape in_shape = {1, 2, 4};
        size_t axis = 0;

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
            auto reshape = ngraph::op::util::reshapeTo(split, shape);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(reshape, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(split);

            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        {
            auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto split = ngraph::builder::makeSplit(input, ngraph::element::i64, 1, axis);
            auto reshape = ngraph::op::util::reshapeTo(split, shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);

            auto read_value = std::make_shared<ngraph::opset3::ReadValue>(copy, "variable_id");
            auto assign = std::make_shared<ngraph::opset3::Assign>(read_value, "variable_id");
            assign->add_control_dependency(read_value);
            auto result = std::make_shared<ngraph::opset8::Result>(split);


            ngraph::ParameterVector params = {input};
            ngraph::ResultVector results = {result};
            ngraph::SinkVector sinks = {assign};
            ref_func = std::make_shared<ngraph::Function>(results, sinks, params);
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeMemoryLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerCropConcatTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        size_t axis = 0;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
            auto concatInput = ngraph::OutputVector{crop};
            auto concat = std::make_shared<ngraph::opset8::Concat>(concatInput, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);

            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);

            auto copy = std::make_shared<ov::intel_gna::op::Copy>(crop);
            auto concatInput = ngraph::OutputVector{copy};

            auto concat = std::make_shared<ngraph::opset8::Concat>(concatInput, axis);
            auto result = std::make_shared<ngraph::opset8::Result>(concat);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Concat");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::InsertCopyBeforeConcatLayer>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerNonfuncTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "nonfunc");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
            auto reshape = ngraph::op::util::reshapeTo(copy, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "nonfunc");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleNonComputationalSubgraphs>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerNonfuncTwoSubgraphsTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape1 = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(params, shape);
            auto result1 = std::make_shared<ngraph::opset8::Result>(reshape1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(reshape2);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                        ngraph::ParameterVector{params},
                                                        "nonfunc");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
            auto reshape1 = ngraph::op::util::reshapeTo(copy, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(copy, shape);
            auto result1 = std::make_shared<ngraph::opset8::Result>(reshape1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(reshape2);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                            ngraph::ParameterVector{params},
                                                            "nonfunc");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleNonComputationalSubgraphs>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerNonfuncTwoResultsTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto result1 = std::make_shared<ngraph::opset8::Result>(reshape);
            auto result2 = std::make_shared<ngraph::opset8::Result>(reshape);
            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                        ngraph::ParameterVector{params},
                                                        "nonfunc");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
            auto reshape = ngraph::op::util::reshapeTo(copy, shape);
            auto result1 = std::make_shared<ngraph::opset8::Result>(reshape);
            auto result2 = std::make_shared<ngraph::opset8::Result>(reshape);
            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                            ngraph::ParameterVector{params},
                                                            "nonfunc");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleNonComputationalSubgraphs>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerNFLBranchTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto reshape2 = ngraph::op::util::reshapeTo(reshape, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

            auto relu = std::make_shared<ngraph::opset8::Relu>(reshape);
            auto result_relu = std::make_shared<ngraph::opset8::Result>(relu);

            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                        ngraph::ParameterVector{params},
                                                        "nonfunc");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);
            auto reshape2 = ngraph::op::util::reshapeTo(copy, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape2);

            auto relu = std::make_shared<ngraph::opset8::Relu>(reshape);
            auto result_relu = std::make_shared<ngraph::opset8::Result>(relu);

            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                            ngraph::ParameterVector{params},
                                                            "nonfunc");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleNonComputationalSubgraphs>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerNFLvsFLSubgraphTest) {
        std::shared_ptr<ngraph::Function> func, ref_func;
        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};
        ngraph::Shape shape         = {1, 1, 2, 4};
        ngraph::Shape in_shape      = {1, 2, 4};

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto reshape = ngraph::op::util::reshapeTo(params, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);

            auto relu = std::make_shared<ngraph::opset8::Relu>(params);
            auto reshape2 = ngraph::op::util::reshapeTo(relu, shape);
            auto result_relu = std::make_shared<ngraph::opset8::Result>(reshape2);

            func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                        ngraph::ParameterVector{params},
                                                        "nonfunc");
        }

        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
            auto copy = std::make_shared<ov::intel_gna::op::Copy>(params);
            auto reshape = ngraph::op::util::reshapeTo(copy, shape);
            auto result = std::make_shared<ngraph::opset8::Result>(reshape);

            auto relu = std::make_shared<ngraph::opset8::Relu>(params);
            auto reshape2 = ngraph::op::util::reshapeTo(relu, shape);
            auto result_relu = std::make_shared<ngraph::opset8::Result>(reshape2);

            ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result, result_relu},
                                                            ngraph::ParameterVector{params},
                                                            "nonfunc");
        }

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::HandleNonComputationalSubgraphs>();
        m.run_passes(func);

        ASSERT_NO_THROW(check_rt_info(func));

        auto result = compare_functions(func, ref_func);
        ASSERT_TRUE(result.first);
}

TEST(TransformationTests, InsertCopyLayerCropNFLConcatTest) {
    std::shared_ptr<ngraph::Function> func, ref_func;
    size_t axis = 0;
    std::vector<int64_t> axes   = {0, 1, 2, 3};
    std::vector<int64_t> dim    = {1, 1, 2, 2};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    ngraph::Shape shape1        = {1, 1, 2, 4};
    ngraph::Shape shape2        = {1, 1, 2, 2};
    ngraph::Shape in_shape      = {1, 2, 4};

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ngraph::op::util::reshapeTo(params, shape1);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
        auto reshape2 = ngraph::op::util::reshapeTo(crop, shape2);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{reshape2}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                    ngraph::ParameterVector{params},
                                                    "Concat");
    }

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, in_shape);
        auto reshape1 = ngraph::op::util::reshapeTo(params, shape1);
        auto crop = std::make_shared<ngraph::op::CropIE>(reshape1, axes, dim, offset);
        auto reshape2 = ngraph::op::util::reshapeTo(crop, shape2);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape2);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                    ngraph::ParameterVector{params},
                                                    "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<GNAPluginNS::InsertCopyBeforeConcatLayer>();
    m.run_passes(func);


    ASSERT_NO_THROW(check_rt_info(func));

    auto result = compare_functions(func, ref_func);
    ASSERT_TRUE(result.first);
}


TEST(TransformationTests, InsertCopyLayerSplitNFLConcatTest) {
    std::shared_ptr<ngraph::Function> func, ref_func;
    ngraph::Shape input_shape{1, 2, 4};
    ngraph::Shape shape{1, 1, 2, 4};
    size_t axis = 0;

    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, 1, axis);
        auto reshape = ngraph::op::util::reshapeTo(split->output(0), shape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{reshape}, axis);
        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                    ngraph::ParameterVector{params},
                                                    "Concat");
    }
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, input_shape);
        auto split = ngraph::builder::makeSplit(params, ngraph::element::i64, 1, axis);
        auto reshape = ngraph::op::util::reshapeTo(split->output(0), shape);
        auto copy = std::make_shared<ov::intel_gna::op::Copy>(reshape);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{copy}, axis);

        auto result = std::make_shared<ngraph::opset8::Result>(concat);
        ref_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Concat");
    }

    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<GNAPluginNS::InsertCopyBeforeConcatLayer>();
    m.run_passes(func);


    ASSERT_NO_THROW(check_rt_info(func));

    auto result = compare_functions(func, ref_func);
    ASSERT_TRUE(result.first);
}

const size_t axis = 0;
const std::vector<size_t> inputCounts = {1, 64, 128, 256};

TEST_P(InsertCopyLayerConcatTest, CompareWithRefs) {
    Run();
}

TEST_P(InsertCopyLayerSplitConcatTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertCopyLayerConcatTest,
                         ::testing::Combine(
                                ::testing::Values(axis),
                                ::testing::ValuesIn(inputCounts)),
                         InsertCopyLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests, InsertCopyLayerSplitConcatTest,
                         ::testing::Combine(
                                ::testing::Values(axis),
                                ::testing::ValuesIn(inputCounts)),
                         InsertCopyLayerTest::getTestCaseName);

} // namespace testing
