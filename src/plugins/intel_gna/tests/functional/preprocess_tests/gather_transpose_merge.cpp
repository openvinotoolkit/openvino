// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::opset10;
using namespace ov::test;

namespace gather_transpose_merge_test {

namespace {

inline std::vector<size_t> make_indexes(size_t size) {
    std::vector<size_t> indexes(size);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::reverse(indexes.begin(), indexes.end());
    return indexes;
}

inline std::vector<size_t> make_transpose_order(const std::vector<size_t>& input_shape) {
    std::vector<size_t> transpose_order;
    switch (input_shape.size()) {
    case 2:
        transpose_order = {1, 0};
        break;
    case 3:
        transpose_order = {0, 2, 1};
        break;
    case 4:
        transpose_order = {0, 2, 3, 1};
        break;
    default:
        break;
    }

    return transpose_order;
}

}  // namespace

typedef std::tuple<std::vector<size_t>,                 // Input shape
                   ov::element::Type,                   // Net precision
                   std::string,                         // Device name
                   std::map<std::string, std::string>,  // Configuration
                   std::map<std::string, std::string>   // Additional Configuration
                   >
    GatherTransposeMergeTestParams;

class GatherTransposeMergeTest : public testing::WithParamInterface<GatherTransposeMergeTestParams>,
                                 virtual public SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<GatherTransposeMergeTestParams>& obj) {
        std::vector<size_t> input_shape;
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf, conf_ext;

        std::tie(input_shape, net_type, target_device, conf, conf_ext) = obj.param;
        for (auto& conf_item : conf_ext) {
            conf[conf_item.first] = conf_item.second;
        }

        std::ostringstream result;
        result << "Shape=" << ov::test::utils::vec2str(input_shape) << "_";
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
        std::tie(m_input_shape, m_net_type, targetDevice, conf, conf_ext) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({m_input_shape, m_input_shape});
        configuration.insert(conf.begin(), conf.end());
        for (auto& conf_item : conf_ext) {
            configuration[conf_item.first] = conf_item.second;
        }
        init_input_shapes(input_shapes);
    }

    void init_test_model();

    ov::element::Type m_net_type;
    std::vector<size_t> m_input_shape;
};

class TransposeGatherTest : public GatherTransposeMergeTest {
protected:
    void init_test_model() {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        std::vector<size_t> transpose_order = make_transpose_order(m_input_shape);
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(params[0], transpose_const);

        std::vector<int8_t> shape_in = {1, -1};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(transpose_node, reshape_in_const, false);

        const std::vector<size_t> gather_ids = make_indexes(ov::shape_size(m_input_shape));
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_out = transpose_node->get_output_shape(0);
        auto reshape_out_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_out_node = std::make_shared<Reshape>(gather_node, reshape_out_const, false);

        ov::ResultVector results{std::make_shared<Result>(reshape_out_node)};
        function = std::make_shared<ov::Model>(results, params, "gather_transpose_merge_test");
    }
};

class GatherTransposeTest : public GatherTransposeMergeTest {
protected:
    void init_test_model() {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        std::vector<int8_t> shape_in = {1, -1};
        auto reshape_in_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_in.size()}, shape_in);
        auto reshape_in_node = std::make_shared<Reshape>(params[0], reshape_in_const, false);

        const std::vector<size_t> gather_ids = make_indexes(ov::shape_size(m_input_shape));
        auto gather_const_ids = Constant::create(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        const size_t gather_axis = 1;
        auto gather_const_axis = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(reshape_in_node, gather_const_ids, gather_const_axis);

        ov::Shape shape_middle = m_input_shape;
        auto reshape_middle_const =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{shape_middle.size()}, shape_middle);
        auto reshape_middle_node = std::make_shared<Reshape>(gather_node, reshape_middle_const, false);

        std::vector<size_t> transpose_order = make_transpose_order(m_input_shape);
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(reshape_middle_node, transpose_const);

        ov::ResultVector results{std::make_shared<Result>(transpose_node)};
        function = std::make_shared<ov::Model>(results, params, "gather_transpose_merge_test");
    }
};

TEST_P(TransposeGatherTest, CompareWithRefs) {
    init_test_model();
    run();
}

TEST_P(GatherTransposeTest, CompareWithRefs) {
    init_test_model();
    run();
}

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<std::map<std::string, std::string>> target_configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<std::vector<size_t>> input_shapes = {{16, 64}, {1, 16, 64}, {1, 8, 16, 64}};

const ov::element::TypeVector input_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_merge_transpose_gather,
                         TransposeGatherTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs)),
                         TransposeGatherTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_merge_transpose_gather,
                         GatherTransposeTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs)),
                         GatherTransposeTest::get_test_case_name);

}  // namespace gather_transpose_merge_test
