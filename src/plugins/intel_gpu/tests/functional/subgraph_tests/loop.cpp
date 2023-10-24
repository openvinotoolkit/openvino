// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/utils/compare_results.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <transformations/control_flow/unroll_tensor_iterator.hpp>

using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using DynamicShapeLoopParams = typename std::tuple<
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InputShape,
        InferenceEngine::Precision,
        std::string,
        ov::AnyMap
        >;

/**
 * Test case with Dynamic SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class DynamicShapeLoopTest : public testing::WithParamInterface<DynamicShapeLoopParams>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicShapeLoopParams> &obj) {
        bool static_iter_num;
        bool static_continue_cond;
        int64_t max_iter_num;
        int64_t dynamic_exit;
        int64_t axis;
        int64_t start_value;
        InputShape data_shapes;
        InferenceEngine::Precision data_prc;
        std::string targetDevice;
        auto args_pack = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
        ov::Any configuration;
        std::tie(
            static_continue_cond,
            args_pack,
            start_value,
            data_shapes,
            data_prc,
            targetDevice,
            configuration) = obj.param;

        std::ostringstream result;
        result << "static_iter_num=" << std::to_string(static_iter_num) << "_";
        result << "static_continue_cond=" << std::to_string(static_continue_cond) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "dynamic_exit=" << std::to_string(dynamic_exit) << "_";
        result << "axis=" << std::to_string(axis) << "_";
        result << "start_value=" << std::to_string(start_value) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        for (size_t i = 0lu; i < data_shapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(data_shapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "netPRC=" << data_prc << "_";
        result << "targetDevice=" << targetDevice << "_";

        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

private:
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InputShape data_shapes;
    InferenceEngine::Precision data_prc;

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        auto args_pack = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
        std::tie(
            static_continue_cond,
            args_pack,
            start_value,
            data_shapes,
            data_prc,
            targetDevice,
            configuration) = GetParam();

        const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(data_prc);
        const auto inputShape = data_shapes.first;
        const auto scalarShape = ngraph::Shape{};
        init_input_shapes({data_shapes});

        ngraph::ParameterVector params{};
        auto cond_input_create = [&params] (ngraph::element::Type prc, const ov::PartialShape &shape, int value = 0, bool is_static = false)
                -> std::shared_ptr<ngraph::Node> {
            if (is_static)
                return std::make_shared<ngraph::opset5::Constant>(prc, shape.to_shape(), value);

            auto input = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
            params.push_back(input);
            return input;
        };

        auto start = cond_input_create(prc, inputShape);
        start->set_friendly_name("start");
        auto count = cond_input_create(ngraph::element::i64, scalarShape, max_iter_num, static_iter_num);
        count->set_friendly_name("count");
        auto skip  = cond_input_create(ngraph::element::boolean, scalarShape, true, static_continue_cond);
        skip->set_friendly_name("skip");

        //
        //      count skip  start         count skip      start
        //                  /                             /
        //          ___*___*____           __________*___*____       | idx  | data | out |
        //         |  idx  in   |         | ex_val  idx  in   |      |  0   |  7   |  7  |
        //         |   |  /     |         |   |   /  |  /     |      |  1   |  7   |  8  |
        //         |   add      |         |   less   add      |      |  2   |  8   |  10 |
        //         |   |   true |         |    |     |        |      |  3   |  10  |  13 |
        //         |   |    |   |         |    |     |        |       ~~~~~  * * *  ~~~~~
        //         |  out  cnd  |         |   cnd   out       |
        //         |___*____*___|         |____*_____*________|
        //           Full loop              Dynamic exit loop
        //           n_iter = count         n_iter = ex_val
        //
        auto b_indx = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{});
        b_indx->set_friendly_name("body_index");
        auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, inputShape);
        b_data->set_friendly_name("body_data");
        auto b_indx_cast = std::make_shared<ngraph::opset5::Convert>(b_indx, prc);
        b_indx_cast->set_friendly_name("body_index_cast");
        auto b_add  = std::make_shared<ngraph::opset5::Add>(b_data, b_indx_cast);
        b_add->set_friendly_name("body_addition");

        std::shared_ptr<ngraph::Node> b_cond;
        if (dynamic_exit == -1) {
            b_cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
            b_cond->set_friendly_name("body_condition");
        } else {
            auto b_exit_value = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, dynamic_exit);
            b_exit_value->set_friendly_name("body_exit_value");
            b_cond = std::make_shared<ngraph::opset5::Less>(b_indx, b_exit_value);
            b_cond->set_friendly_name("body_condition_with_exit_value");
        }

        auto body = std::make_shared<ngraph::Function>(
                ngraph::OutputVector    {b_cond, b_add},    // TODO: check with reverse
                ngraph::ParameterVector {b_indx, b_data});  // TODO: check with reverse
        body->set_friendly_name("body_network");

        auto loop = std::make_shared<ngraph::opset5::Loop>(count, skip);
        loop->set_friendly_name("loop");
        loop->set_function(body);
        loop->set_special_body_ports({0, 0});
        loop->set_merged_input(b_data, start, b_add);
        if (axis == -1)
            loop->get_iter_value(b_add, -1);
        else
            loop->get_concatenated_slices(b_add, 0, 1, 1, -1, axis);

        function = std::make_shared<ngraph::Function>(
                ngraph::OutputVector {loop},
                params);
        function->set_friendly_name("outer_body_network");
    }
};


TEST_P(DynamicShapeLoopTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32
};

ov::AnyMap netConfigurations = {
    {GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, PluginConfigParams::NO}
};

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_axis_0 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, 0 },  // n_iter 10, no dynamic exit
};

std::vector<InputShape> inputs_0 = {
    InputShape(ov::PartialShape({1, -1, 2}), {{1, 4, 2}, {1, 5, 2}, {1, 10, 2}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_0, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_axis_0),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_0),
                        /* data_prc */ testing::ValuesIn(netPrecisions),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* configuration */ testing::Values<ov::AnyMap>(netConfigurations)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_1 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, no dynamic exit
};

std::vector<InputShape> inputs_1 = {
    InputShape(ov::PartialShape({-1, 1, 4, -1}), {{2, 1, 4, 10}, {3, 1, 4, 14}, {6, 1, 4, 16}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_1, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_1),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_1),
                        /* data_prc */ testing::ValuesIn(netPrecisions),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* configuration */ testing::Values<ov::AnyMap>(netConfigurations)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_2 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1,  2 },  // n_iter 10, no dynamic exit
};

std::vector<InputShape> inputs_2 = {
    InputShape(ov::PartialShape({-1, -1, 1, 6}), {{2, 4, 1, 6}, {10, 40, 1, 6}, {12, 16, 1, 6}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_2, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_2),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_2),
                        /* data_prc */ testing::ValuesIn(netPrecisions),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* configuration */ testing::Values<ov::AnyMap>(netConfigurations)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_no_auto_concat {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, -1 },  // n_iter 5, no dynamic exit
};

std::vector<InputShape> inputs_no_auto_concat = {
    InputShape(ov::PartialShape({-1, 1, 6}), {{2, 1, 6}, {10, 1, 6}, {12, 1, 6}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_no_auto_concat, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_no_auto_concat),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_no_auto_concat),
                        /* data_prc */ testing::ValuesIn(netPrecisions),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* configuration */ testing::Values<ov::AnyMap>(netConfigurations)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_dynamic_exit {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3,  -1 },  // n_iter 3, dynamic exit on 3
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  7,   1 },  // n_iter 5, dynamic exit not reached
    std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5,  -1 },  // n_iter 5, inf loop with dynamic exit on 5
};

std::vector<InputShape> inputs_dynamic_exit = {
    InputShape(ov::PartialShape({-1, 1, 2}), {{4, 1, 2}, {10, 1, 2}, {12, 1, 2}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_dynamic_exit, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_dynamic_exit),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_dynamic_exit),
                        /* data_prc */ testing::ValuesIn(netPrecisions),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* configuration */ testing::Values<ov::AnyMap>(netConfigurations)),
                        DynamicShapeLoopTest::getTestCaseName);

} // namespace GPULayerTestsDefinitions