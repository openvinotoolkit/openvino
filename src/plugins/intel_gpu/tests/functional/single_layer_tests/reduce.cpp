// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

using ov::test::InputShape;
using ReduceInputParams = std::tuple<
                            ov::Shape,                // Input shapes
                            ov::element::Type,        // Input precision
                            std::vector<int64_t>,     // Reduce Axes
                            bool>;                    // Keep dims


class ReduceSumSqueezeTest : public testing::WithParamInterface<ReduceInputParams>,
          virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceInputParams>& obj) {
        const auto& [input_shape, input_precisions, reduce_axes, keep_dims] = obj.param;

        std::ostringstream result;
        result << "IS=ReduceSum_";
        result << ov::test::utils::vec2str(input_shape) << "_";
        result << "reduce_axes=" << ov::test::utils::vec2str(reduce_axes) << "_";
        result << "keep_dims=" << keep_dims << "_";
        result << "input_precision=" << input_precisions;
        return result.str();
    }

protected:
    ov::Shape input_shape;
    std::vector<int64_t> reduce_axes;
    bool keep_dims;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        core->set_property(targetDevice, ov::enable_profiling(true));

        const auto& [_input_shape, input_precision, _reduce_axes, _keep_dims] = GetParam();
        input_shape = _input_shape;
        reduce_axes = _reduce_axes;
        keep_dims = _keep_dims;

        ov::Shape add_val_shape = {input_shape.begin(), input_shape.end() - 1};
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -10;
        in_data.range = 10;
        auto add_val_tensor = ov::test::utils::create_and_fill_tensor(input_precision, add_val_shape, in_data);

        auto input_node = std::make_shared<ov::op::v0::Parameter>(input_precision, input_shape);
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {1}, reduce_axes);
        auto reduce_node = std::make_shared<ov::op::v1::ReduceSum>(input_node, axes_node, keep_dims);
        reduce_node->set_friendly_name("ReduceSum");
        auto add_val_node = std::make_shared<ov::op::v0::Constant>(add_val_tensor);
        auto add_node = std::make_shared<ov::op::v1::Add>(reduce_node, add_val_node);
        auto mul_val_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 1.5f);
        auto mul_node = std::make_shared<ov::op::v1::Multiply>(add_node, mul_val_node);
        auto result = std::make_shared<ov::op::v0::Result>(mul_node);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_node}, "input");
    }

    void run() override {
        ov::test::SubgraphBaseStaticTest::run();

        const auto& [input_shape, _ignore, _ignore1, _ignore2] = GetParam();
        std::ignore = _ignore;
        std::ignore = _ignore1;
        std::ignore = _ignore2;
        auto profile_info = inferRequest.get_profiling_info();
        auto num_executed = std::count_if(profile_info.begin(), profile_info.end(),
            [](const ov::ProfilingInfo& p) { return p.status == ov::ProfilingInfo::Status::EXECUTED; });
        // This ensures that primitive_fusing_through does not happen across "dimension-change-barrier" in reduce
        ASSERT_EQ(num_executed, 3);
    }
};

TEST_P(ReduceSumSqueezeTest, Inference) {
    run();
}

const std::vector<ov::Shape> input_shapes = {{{1, 2, 3, 2, 4}}};
const std::vector<ov::element::Type> input_prec = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(
    smoke_ReduceSumSqueezeTest,
    ReduceSumSqueezeTest,
    ::testing::Combine(::testing::ValuesIn(input_shapes),
                       ::testing::ValuesIn(input_prec),
                       ::testing::Values(std::vector<int64_t>{4}),
                       ::testing::Values(false)),
    ReduceSumSqueezeTest::getTestCaseName);

using ReduceParams = std::tuple<ov::Shape,             // Input shape
                                ov::element::Type,     // Input precision
                                std::vector<int64_t>,  // Reduce axes
                                bool>;                 // Keep dims

template <typename ReduceOp>
class ReduceTest : public testing::WithParamInterface<ReduceParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceParams>& obj) {
        const auto& [input_shape, precision, axes, keep_dims] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
        result << "axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "keep_dims=" << keep_dims << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, precision, axes, keep_dims] = GetParam();
        auto input = std::make_shared<ov::op::v0::Parameter>(precision, input_shape);
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
        auto reduce = std::make_shared<ReduceOp>(input, axes_node, keep_dims);
        auto result = std::make_shared<ov::op::v0::Result>(reduce);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "Reduce");
    }
};

using ReduceMeanTest = ReduceTest<ov::op::v1::ReduceMean>;
using ReduceMaxTest = ReduceTest<ov::op::v1::ReduceMax>;
using ReduceMinTest = ReduceTest<ov::op::v1::ReduceMin>;
using ReduceProdTest = ReduceTest<ov::op::v1::ReduceProd>;
using ReduceSumTest = ReduceTest<ov::op::v1::ReduceSum>;

TEST_P(ReduceMeanTest, Inference) { run(); }
TEST_P(ReduceMaxTest, Inference) { run(); }
TEST_P(ReduceMinTest, Inference) { run(); }
TEST_P(ReduceProdTest, Inference) { run(); }
TEST_P(ReduceSumTest, Inference) { run(); }

const auto reduce_test_params = ::testing::Combine(::testing::Values(ov::Shape{1, 24, 1024, 64}),
                                                   ::testing::Values(ov::element::f32),
                                                   ::testing::Values(std::vector<int64_t>{3}),
                                                   ::testing::Values(true));
// ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(smoke_ReduceMeanTest, ReduceMeanTest, reduce_test_params, ReduceMeanTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ReduceMaxTest, ReduceMaxTest, reduce_test_params, ReduceMaxTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ReduceMinTest, ReduceMinTest, reduce_test_params, ReduceMinTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ReduceProdTest, ReduceProdTest, reduce_test_params, ReduceProdTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ReduceSumTest, ReduceSumTest, reduce_test_params, ReduceSumTest::getTestCaseName);

}  // namespace
