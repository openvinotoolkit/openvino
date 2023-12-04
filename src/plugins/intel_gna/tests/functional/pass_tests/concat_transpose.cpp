// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/opsets/opset12.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::opset12;
using namespace ov::test;

namespace TransposesConcatTests {

typedef std::tuple<ov::element::Type,                   // Net precision
                   std::string,                         // Device name
                   std::map<std::string, std::string>,  // Configuration
                   std::map<std::string, std::string>   // Additional Configuration
                   >
    TransposesConcatTestParamsSet;

class TransposesConcatTest : public testing::WithParamInterface<TransposesConcatTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<TransposesConcatTestParamsSet>& obj) {
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf, conf_ext;

        std::tie(net_type, target_device, conf, conf_ext) = obj.param;
        for (auto& conf_item : conf_ext) {
            conf[conf_item.first] = conf_item.second;
        }

        std::ostringstream result;
        result << "netPRC=" << net_type << "_";
        result << "trgDev=" << target_device;
        for (auto const& conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();
        std::map<std::string, std::string> conf, conf_ext;
        std::tie(m_net_type, targetDevice, conf, conf_ext) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({{10, 1}});
        configuration.insert(conf.begin(), conf.end());
        for (auto& conf_item : conf_ext) {
            configuration[conf_item.first] = conf_item.second;
        }
        init_input_shapes(input_shapes);
    }

    void init_test_model() {
        std::vector<std::vector<size_t>> input_shapes = {{10, 1}};

        ov::ParameterVector params;
        for (auto&& shape : input_shapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(shape)));
        }
        std::vector<size_t> shape_1 = {10, 128};
        std::vector<size_t> shape_2 = {10, 192};
        std::vector<size_t> shape_3 = {10, 256};

        std::shared_ptr<ov::Node> input_node = params[0];

        std::vector<float> mulConstWeights(1 * 576);
        std::iota(mulConstWeights.begin(), mulConstWeights.end(), 0.1f);
        auto constMul1 = ngraph::builder::makeConstant<float>(m_net_type, ov::Shape{1, 576}, mulConstWeights);
        auto matmul1 = std::make_shared<ov::opset10::MatMul>(input_node, constMul1, false, false);

        auto split_axis = std::make_shared<Constant>(ov::element::u8, ov::Shape{1}, std::vector<uint8_t>{1});
        auto split_slices =
            std::make_shared<Constant>(ov::element::u32, ov::Shape{3}, std::vector<uint32_t>{128, 192, 256});
        auto split_node = std::make_shared<VariadicSplit>(matmul1, split_axis, split_slices);

        std::vector<size_t> transpose_order = {1, 0};
        auto transpose_const_1 =
            std::make_shared<Constant>(ov::element::u8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_const_2 =
            std::make_shared<Constant>(ov::element::u8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_const_3 =
            std::make_shared<Constant>(ov::element::u8, ov::Shape{transpose_order.size()}, transpose_order);

        auto transpose_node_1 = std::make_shared<Transpose>(split_node->output(0), transpose_const_1);
        auto transpose_node_2 = std::make_shared<Transpose>(split_node->output(1), transpose_const_2);
        auto transpose_node_3 = std::make_shared<Transpose>(split_node->output(2), transpose_const_3);

        const int axis = 0;
        auto concat_node =
            std::make_shared<Concat>(ov::OutputVector{transpose_node_1, transpose_node_2, transpose_node_3}, axis);

        std::vector<size_t> reshape_pattern = {1, 2, 5, 576};
        auto reshape_const =
            std::make_shared<Constant>(ov::element::u16, ov::Shape{reshape_pattern.size()}, reshape_pattern);
        auto reshape_node = std::make_shared<Reshape>(concat_node, reshape_const, false);

        ov::ResultVector results{std::make_shared<Result>(reshape_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }

    ov::element::Type m_net_type;
    std::vector<size_t> m_input_shape;
};

TEST_P(TransposesConcatTest, CompareWithRefs) {
    init_test_model();
    run();
}

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<std::map<std::string, std::string>> target_configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_1_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const ov::element::TypeVector input_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_transposes_concat,
                         TransposesConcatTest,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs)),
                         TransposesConcatTest::get_test_case_name);

}  // namespace TransposesConcatTests
