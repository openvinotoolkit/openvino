// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/opsets/opset10.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::opset10;
using namespace ov::test;

namespace unsupported_transpose_tests {

typedef std::tuple<ov::element::Type,                   // Net precision
                   ov::Shape,                           // Transpose shape
                   std::string,                         // Device name
                   std::map<std::string, std::string>,  // Configuration
                   std::map<std::string, std::string>   // Additional Configuration
                   >
    UnsupportedTransposeTestParams;

class UnsupportedTransposeTest : public testing::WithParamInterface<UnsupportedTransposeTestParams>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<UnsupportedTransposeTestParams>& obj) {
        ov::element::Type net_type, in_type, out_type;
        ov::Shape tr_shape;
        std::string target_device;
        std::map<std::string, std::string> conf, conf_ext;

        std::tie(net_type, tr_shape, target_device, conf, conf_ext) = obj.param;
        for (auto& conf_item : conf_ext) {
            conf[conf_item.first] = conf_item.second;
        }

        std::ostringstream result;
        result << "netPRC=" << net_type << "_";
        result << "shape=" << tr_shape << "_";
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
        std::tie(m_net_type, m_test_shape, targetDevice, conf, conf_ext) = this->GetParam();
        // set input shapes
        m_input_shapes.push_back({1, m_test_shape[1]});
        m_input_shapes.push_back({m_test_shape[0] - 1, m_test_shape[1]});

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation(m_input_shapes);
        configuration.insert(conf.begin(), conf.end());
        for (auto& conf_item : conf_ext) {
            configuration[conf_item.first] = conf_item.second;
        }
        init_input_shapes(input_shapes);
    }

    void init_test_model() {
        ov::ParameterVector params = {};
        for (const auto& shape_in : m_input_shapes) {
            params.push_back(std::make_shared<Parameter>(m_net_type, shape_in));
        }

        const int concat_axis = 0;
        auto concat_node = std::make_shared<Concat>(ov::OutputVector{params[0], params[1]}, concat_axis);

        std::vector<size_t> transpose_order = {1, 0};
        auto transpose_const =
            std::make_shared<Constant>(ov::element::u8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(concat_node, transpose_const);

        auto split_axis = std::make_shared<Constant>(ov::element::u8, ov::Shape{1}, std::vector<uint8_t>{0});
        auto split_slices =
            std::make_shared<Constant>(ov::element::u32, ov::Shape{2}, std::vector<uint32_t>{m_test_shape[1] - 1, 1});
        auto split_node = std::make_shared<VariadicSplit>(transpose_node, split_axis, split_slices);

        ov::ResultVector results{std::make_shared<Result>(split_node->output(0)),
                                 std::make_shared<Result>(split_node->output(1))};
        function = std::make_shared<ov::Model>(results, params, "unsupported_transpose");
    }

    ov::element::Type m_net_type;
    ov::Shape m_test_shape;
    std::vector<ov::Shape> m_input_shapes;
};

TEST_P(UnsupportedTransposeTest, CompareWithRefs) {
    init_test_model();
    run();
}

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<std::map<std::string, std::string>> target_configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const ov::element::TypeVector input_precisions = {ov::element::f32};

const std::vector<ov::Shape> transpose_shapes = {{67, 64}, {64, 67}, {64, 64}};

INSTANTIATE_TEST_SUITE_P(smoke_transposes_concat,
                         UnsupportedTransposeTest,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(transpose_shapes),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs)),
                         UnsupportedTransposeTest::get_test_case_name);

}  // namespace unsupported_transpose_tests
