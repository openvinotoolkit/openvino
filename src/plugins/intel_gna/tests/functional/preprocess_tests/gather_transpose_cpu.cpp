// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "openvino/opsets/opset9.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov::opset9;
using namespace ov::test;

namespace LayerTestsDefinitions {

namespace {

std::vector<size_t> MakeIndexes(size_t size) {
    std::vector<size_t> indexes(size);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::reverse(indexes.begin(), indexes.end());
    return indexes;
}

std::vector<size_t> MakeTransposeOrder(const std::vector<size_t>& input_shape) {
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

typedef std::tuple<std::vector<size_t>,                // Input shape
                   ov::element::Type,                  // Net precision
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Configuration
                   >
    preprocessTestParamsSet;

class PreprocessBaseTest : public testing::WithParamInterface<preprocessTestParamsSet>,
                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessTestParamsSet>& obj) {
        std::vector<size_t> input_shape;
        ov::element::Type net_type, in_type, out_type;
        std::string target_device;
        std::map<std::string, std::string> conf;

        std::tie(input_shape, net_type, target_device, conf) = obj.param;

        std::ostringstream result;
        result << "Shape=" << CommonTestUtils::vec2str(input_shape) << "_";
        result << "netPRC=" << net_type << "_";
        result << "trgDev=" << target_device;
        for (auto const& conf_i : conf) {
            result << "_configItem=" << conf_i.first.c_str() << "_" << conf_i.second.c_str();
        }
        return result.str();
    }

protected:
    void InitTestModel();
    void SetUp() override {
        abs_threshold = std::numeric_limits<int32_t>::max();
        rel_threshold = std::numeric_limits<int32_t>::max();
        std::tie(m_input_shape, m_net_type, targetDevice, m_conf) = this->GetParam();

        std::vector<InputShape> input_shapes = static_shapes_to_test_representation({m_input_shape, m_input_shape});
        configuration.insert(m_conf.begin(), m_conf.end());

        init_input_shapes(input_shapes);
    }

    ov::element::Type m_net_type;
    std::vector<size_t> m_input_shape;
    std::map<std::string, std::string> m_conf;
};

class RemoveGatherInput : public PreprocessBaseTest {
protected:
    void InitTestModel() {
        auto params = ngraph::builder::makeParams(m_net_type, {m_input_shape});
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        const std::vector<size_t> indexes = MakeIndexes(input_shape_size);
        auto gather_indexes = Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = m_input_shape.size() - 1;
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(params[0], gather_indexes, gather_axis_const);

        auto mul_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(gather_node, mul_input_const);

        auto add_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        ov::ResultVector results{std::make_shared<Result>(add_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveGatherOutput : public PreprocessBaseTest {
protected:
    void InitTestModel() {
        auto params = ngraph::builder::makeParams(m_net_type, {m_input_shape});

        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto mul_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(params[0], mul_input_const);

        auto add_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        const std::vector<size_t> indexes = MakeIndexes(input_shape_size);
        auto gather_indexes = Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = m_input_shape.size() - 1;
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<Gather>(add_node, gather_indexes, gather_axis_const);

        ov::ResultVector results{std::make_shared<Result>(gather_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveTransposeInput : public PreprocessBaseTest {
protected:
    void InitTestModel() {
        std::vector<size_t> transpose_order = MakeTransposeOrder(m_input_shape);
        std::vector<size_t> transpose_shape;
        for (size_t i : transpose_order) {
            transpose_shape.push_back(m_input_shape[i]);
        }

        auto params = ngraph::builder::makeParams(m_net_type, {m_input_shape});
        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(params[0], transpose_const);

        auto mul_input_const = Constant::create(m_net_type,
                                                transpose_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(transpose_node, mul_input_const);

        auto add_input_const = Constant::create(m_net_type,
                                                transpose_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        ov::ResultVector results{std::make_shared<Result>(add_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

class RemoveTransposeOutput : public PreprocessBaseTest {
protected:
    void InitTestModel() {
        std::vector<size_t> transpose_order = MakeTransposeOrder(m_input_shape);
        std::vector<size_t> transpose_shape;
        for (size_t i : transpose_order) {
            transpose_shape.push_back(m_input_shape[i]);
        }

        auto params = ngraph::builder::makeParams(m_net_type, {m_input_shape});

        const size_t input_shape_size = ov::shape_size(params[0]->get_shape());

        auto mul_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto matmul_node = std::make_shared<Multiply>(params[0], mul_input_const);

        auto add_input_const = Constant::create(m_net_type,
                                                m_input_shape,
                                                CommonTestUtils::generate_float_numbers(input_shape_size, -0.2f, 0.2f));
        auto add_node = std::make_shared<Add>(matmul_node, add_input_const);

        auto transpose_const =
            std::make_shared<Constant>(ov::element::i8, ov::Shape{transpose_order.size()}, transpose_order);
        auto transpose_node = std::make_shared<Transpose>(add_node, transpose_const);

        ov::ResultVector results{std::make_shared<Result>(transpose_node)};
        function = std::make_shared<ov::Model>(results, params, "concat");
    }
};

TEST_P(RemoveTransposeInput, CompareWithRefs) {
    InitTestModel();
    run();
}

TEST_P(RemoveTransposeOutput, CompareWithRefs) {
    InitTestModel();
    run();
}

TEST_P(RemoveGatherInput, CompareWithRefs) {
    InitTestModel();
    run();
}

TEST_P(RemoveGatherOutput, CompareWithRefs) {
    InitTestModel();
    run();
}

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> input_shapes_transpose = {{2, 64}, {1, 2, 64}, {1, 2, 4, 16}};

const std::vector<std::vector<size_t>> input_shapes_gather = {{1, 128}};

const ov::element::TypeVector input_precisions = {ov::element::f32};

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveGatherInput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_gather),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveGatherInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveGatherOutput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_gather),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveGatherInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveTransposeInput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_transpose),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveTransposeInput::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_preprocess,
                         RemoveTransposeOutput,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_transpose),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         RemoveTransposeOutput::getTestCaseName);

}  // namespace LayerTestsDefinitions
