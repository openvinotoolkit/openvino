// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::opset10;
using namespace ov::test;

enum PrePostLayer { NONE, GATHER, TRANSPOSE };

namespace std {
inline std::ostream& operator<<(std::ostream& os, PrePostLayer layer_type) {
    switch (layer_type) {
    case PrePostLayer::GATHER:
        os << "GATHER";
        break;
    case PrePostLayer::TRANSPOSE:
        os << "TRANSPOSE";
        break;
    default:
        os << "NONE";
        break;
    }
    return os;
}
}  // namespace std

namespace LayerTestsDefinitions {

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
                   std::map<std::string, std::string>,  // Additional Configuration
                   PrePostLayer,                        // Pre-processing layer
                   PrePostLayer                         // Post-processing layer
                   >
    preprocessTestParamsSet;

class PrePostProcessBaseTest : public testing::WithParamInterface<preprocessTestParamsSet>,
                               virtual public SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<preprocessTestParamsSet>& obj) {
        std::vector<size_t> input_shape;
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf, conf_ext;
        PrePostLayer pre_layer, post_layer;

        std::tie(input_shape, net_type, target_device, conf, conf_ext, pre_layer, post_layer) = obj.param;
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
        result << "_pre_layer=" << pre_layer;
        result << "_post_layer=" << post_layer;
        return result.str();
    }

protected:
    void SetUp() override {
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();
        std::map<std::string, std::string> conf, conf_ext;
        std::tie(m_input_shape, m_net_type, targetDevice, conf, conf_ext, m_pre_layer, m_post_layer) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({m_input_shape, m_input_shape});
        configuration.insert(conf.begin(), conf.end());
        for (auto& conf_item : conf_ext) {
            configuration[conf_item.first] = conf_item.second;
        }
        init_input_shapes(input_shapes);
    }

    void init_test_model() {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(m_net_type, ov::Shape(m_input_shape))};
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        std::shared_ptr<ov::Node> pre_node = params[0];
        switch (m_pre_layer) {
        case PrePostLayer::GATHER:
            pre_node = make_gather_node(pre_node);
            break;
        case PrePostLayer::TRANSPOSE:
            pre_node = make_transpose_node(pre_node);
            break;
        default:
            break;
        }

        auto mul_input_const = Constant::create(m_net_type,
                                                pre_node->get_shape(),
                                                ov::test::utils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(pre_node, mul_input_const);

        auto add_input_const = Constant::create(m_net_type,
                                                pre_node->get_shape(),
                                                ov::test::utils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        std::shared_ptr<ov::Node> post_node = add_node;
        switch (m_post_layer) {
        case PrePostLayer::GATHER:
            pre_node = make_gather_node(post_node);
            break;
        case PrePostLayer::TRANSPOSE:
            pre_node = make_transpose_node(post_node);
            break;
        default:
            break;
        }

        ov::ResultVector results{std::make_shared<Result>(post_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }

    std::shared_ptr<ov::Node> make_gather_node(std::shared_ptr<ov::Node> node) {
        const std::vector<size_t> indexes = make_indexes(ov::shape_size(m_input_shape));
        auto gather_indexes = Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = m_input_shape.size() - 1;
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(node, gather_indexes, gather_axis_const);

        return gather_node;
    }

    std::shared_ptr<ov::Node> make_transpose_node(std::shared_ptr<ov::Node> node) {
        std::vector<size_t> transpose_order = make_transpose_order(m_input_shape);
        ov::Shape transpose_shape;
        for (size_t i : transpose_order) {
            transpose_shape.push_back(m_input_shape[i]);
        }

        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(node, transpose_const);

        return transpose_node;
    }

    ov::element::Type m_net_type;
    std::vector<size_t> m_input_shape;
    PrePostLayer m_pre_layer = PrePostLayer::NONE;
    PrePostLayer m_post_layer = PrePostLayer::NONE;
};

TEST_P(PrePostProcessBaseTest, CompareWithRefs) {
    init_test_model();
    run();
}

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<std::map<std::string, std::string>> target_configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_1_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
                                                                  {{"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<std::vector<size_t>> input_shapes_transpose = {{2, 64}, {1, 2, 64}, {1, 2, 4, 16}};

const std::vector<std::vector<size_t>> input_shapes_gather = {{1, 128}};

const ov::element::TypeVector input_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_preprocess_gather,
                         PrePostProcessBaseTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_gather),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs),
                                            ::testing::ValuesIn({PrePostLayer::NONE, PrePostLayer::GATHER}),
                                            ::testing::ValuesIn({PrePostLayer::NONE, PrePostLayer::GATHER})),
                         PrePostProcessBaseTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess_transpose,
                         PrePostProcessBaseTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_transpose),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(target_configs),
                                            ::testing::ValuesIn({PrePostLayer::NONE, PrePostLayer::TRANSPOSE}),
                                            ::testing::ValuesIn({PrePostLayer::NONE, PrePostLayer::TRANSPOSE})),
                         PrePostProcessBaseTest::get_test_case_name);

}  // namespace LayerTestsDefinitions
