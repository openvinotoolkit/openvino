// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/loop.hpp"

namespace LayerTestsDefinitions {

    std::string LoopTest::getTestCaseName(const testing::TestParamInfo<LoopParams> &obj) {
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>> inputs;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, inputs, netPrecision,
                 targetDevice) = obj.param;

        std::vector<std::vector<size_t>> inputs_separate;
        std::vector<LOOP_IN_TYPE> types_separate;
        for (auto &el : inputs) {
            inputs_separate.push_back(el.first);
            types_separate.push_back(el.second);
        }
        std::ostringstream result;
        result << "execute_first_iteration" << execute_first_iteration << "_";
        result << "is_body_condition_const=" << is_body_condition_const << "_";
        result << "body_condition=" << body_condition << "_";
        result << "trip_count=" << trip_count << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputs_separate) << "_";
        result << "types=" << CommonTestUtils::vec2str(types_separate) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

    void LoopTest::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        SetRefMode(LayerTestsUtils::IE);
        bool execute_first_iteration;
        bool is_body_condition_const;
        bool body_condition; // works only if is_body_condition_const ==
        int64_t trip_count;
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>> inputs;
        InferenceEngine::Precision netPrecision;
        std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, inputs, netPrecision,
                 targetDevice) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        // That which we iterate over
        std::vector<std::vector<size_t>> inputs_separate;
        std::vector<LOOP_IN_TYPE> types_separate;
        for (auto &el : inputs) {
            inputs_separate.push_back(el.first);
            types_separate.push_back(el.second);
        }
        // Example:
        /*      auto X = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});
        auto Y = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});
        auto M = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{32, 1, 10});*/
        auto params = ngraph::builder::makeParams(ngPrc, inputs_separate);

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ngraph::PartialShape> body_params_shapes(inputs_separate.size(), ngraph::PartialShape::dynamic());
        auto current_iteration = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64, ngraph::Shape{1});

        //Example:
/*      auto Xi = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto Yi = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto M_body = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());*/

        ngraph::ParameterVector body_params;
        for (const auto &pshape : body_params_shapes) {
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, pshape);
            body_params.push_back(paramNode);
        }

        std::shared_ptr<ngraph::Node> body_condition_const;
        if (is_body_condition_const) {
            if (body_condition) {
                body_condition_const = std::make_shared<ngraph::opset5::Constant>(
                        ngraph::element::boolean, ngraph::Shape{1}, true);
            } else {
                body_condition_const = std::make_shared<ngraph::opset5::Constant>(
                        ngraph::element::boolean, ngraph::Shape{1}, false);
            }
        }

        auto trip_count_const =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, trip_count);

        std::shared_ptr<ngraph::Node> exec_condition;
        if (execute_first_iteration) {
            exec_condition = std::make_shared<ngraph::opset5::Constant>(
                    ngraph::element::boolean, ngraph::Shape{1}, true);
        } else {
            exec_condition = std::make_shared<ngraph::opset5::Constant>(
                    ngraph::element::boolean, ngraph::Shape{1}, false);
        }

        // Body
        std::shared_ptr<ngraph::Node> Zo = body_params[0];
        for (int i = 1; i < body_params.size(); ++i) {
            Zo = body_params[i] + Zo;
        }

        // body_params.insert(body_params.begin(), current_iteration);
        auto body = std::make_shared<ngraph::Function>(ngraph::OutputVector{body_condition_const, Zo},
                                                  body_params);

        auto loop = std::make_shared<ngraph::opset5::Loop>(trip_count_const, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

        for (int i = 0; i < body_params.size(); ++i) {
            if (types_separate[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], params[i]);
            } else if (types_separate[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], params[i], Zo);
            }
        }

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition_const, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        auto result0 = std::make_shared<ngraph::op::Result>(out0);
        auto result1 = std::make_shared<ngraph::op::Result>(out1);
        auto result2 = std::make_shared<ngraph::op::Result>(out2);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result0, result1, result2}, params, "loop");
    }


    TEST_P(LoopTest, CompareWithRefs) {
        Run();
    };
}  // namespace LayerTestsDefinitions
