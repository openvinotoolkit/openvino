// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/precision_propagation.hpp"

#include <gtest/gtest.h>
#include <ngraph/pass/constant_folding.hpp>
#include <snippets/pass/propagate_precision.hpp>
#include "common_test_utils/common_utils.hpp"
#include "precision_propagation_function.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
class DummyAdd : public ngraph::opset1::Add {
public:
    OPENVINO_OP("Add", "test::snippets");

    DummyAdd(const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast =
            ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))
        : ngraph::opset1::Add(arg0, arg1, auto_broadcast) {
        constructor_validate_and_infer_types();
    }

    DummyAdd(const ngraph::opset1::Add& add)
        : Add(add.get_input_source_output(0), add.get_input_source_output(1), add.get_autob()) {
        constructor_validate_and_infer_types();
    }

    DummyAdd() = default;

    void validate_and_infer_types() override {
        const auto input_type1 = get_input_element_type(0);
        const auto input_type2 = get_input_element_type(1);

        // TODO: not completed
        const element::Type output_type =
            (input_type1 == element::i8) || (input_type2 == element::i8) ? element::i32 : get_input_element_type(0);

        set_output_type(0, output_type, get_input_partial_shape(0));
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
        // TODO: not completed
        return true;
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
};

class DummyConvertToSnippetsOpset : ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("DummyConvertToSnippetsOpset", "0");
    DummyConvertToSnippetsOpset() {}
    
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override {
        for (const auto& op : m->get_ordered_ops()) {
            if (ngraph::is_type<ngraph::opset1::Add>(op)) {
                auto new_instance = std::make_shared<DummyAdd>(*ngraph::as_type_ptr<ngraph::opset1::Add>(op));
                replace_node(op, new_instance);
                new_instance->validate_and_infer_types();
            }
        }
        return false;
    }
};

class DummyPrecisionPropogationTargetMachine : public DummyTargetMachine {
public:
    DummyPrecisionPropogationTargetMachine() : DummyTargetMachine() {
        auto add_functor = ngraph::snippets::jitters_value {
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            {
                {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
                {InferenceEngine::Precision::I8, InferenceEngine::Precision::I8}
            }
        };

        //auto max_pool_functor = ngraph::snippets::jitters_value {
        //    [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
        //    {{InferenceEngine::Precision::FP32}}
        //};

        auto max_functor = ngraph::snippets::jitters_value{
            [](const std::shared_ptr<ngraph::Node>& n) { return std::make_shared<DummyEmitter>(); },
            {{InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32}}
        };

        jitters[DummyAdd::get_type_info_static()] = add_functor;
        //jitters[op::v1::MaxPool::get_type_info_static()] = max_pool_functor;
        jitters[op::v1::Maximum::get_type_info_static()] = max_functor;
    }
    bool is_supported() const override {
        return true;
    }
    ngraph::snippets::code get_snippet() const override {
        return nullptr;
    }
    size_t get_lanes() const override {
        return 10;
    }
};

} // namespace

std::string PrecisionPropagationTest::getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj) {
    std::pair<Shape, Shape> shapes;
    PrecisionPropagationParamsValues test_values;
    std::tie(shapes, test_values) = obj.param;

    //const auto param = GetParam();
    //const auto shapes = std::get<0>(param);
    //const auto test_values = std::get<1>(param);

    std::ostringstream result;
    result << "IN0_" << shapes.first << "_" << test_values.actual.input_types.first << "_";
    result << "IN1_" << shapes.second << "_" << test_values.actual.input_types.second;
    return result.str();
}

void PrecisionPropagationTest::SetUp() {
    const auto param = GetParam();
    const auto shapes = std::get<0>(param);
    const auto test_values = std::get<1>(param);

    function = PrecisionPropagationFunction::get(test_values.actual.input_types.first,
                                                 shapes.first,
                                                 test_values.actual.input_types.second,
                                                 shapes.second);
    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);

    DummyConvertToSnippetsOpset().run_on_model(function);
    const auto target_machine = std::make_shared<DummyPrecisionPropogationTargetMachine>();
    ngraph::snippets::pass::PropagatePrecision(element::f32, target_machine).run_on_model(function);    
    ngraph::pass::VisualizeTree("svg/test.transformed.svg").run_on_model(function);

    function_ref = PrecisionPropagationFunction::get(test_values.actual.input_types.first,
                                                     shapes.first, 
                                                     test_values.actual.input_types.second,
                                                     shapes.second);
    ngraph::pass::VisualizeTree("svg/test.reference.svg").run_on_model(function_ref);
}

TEST_P(PrecisionPropagationTest, CompareFunctions) {
    // TODO: why I need it?
    disable_rt_info_check();
    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

namespace PrecisionPropagationTestInstantiation {

std::vector<std::pair<Shape, Shape>> shapes {
    {{1, 3, 16, 16}, {1, 3, 16, 16}}
};

//std::vector<std::pair<element::Type, element::Type>> types{
//    //{element::f32, element::f32}, 
//    {element::i8, element::i8}
//};

std::vector<PrecisionPropagationParamsValues> test_cases {
    {
        // Actual
        {{element::i8, element::i8}, {}, {}}, 
        // Expected
        {}
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_PrecisionPropagationTest, 
    PrecisionPropagationTest,
    ::testing::Combine(
        ::testing::ValuesIn(shapes), 
        ::testing::ValuesIn(test_cases)),
    PrecisionPropagationTest::getTestCaseName);

} // namespace PrecisionPropagationTestInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov