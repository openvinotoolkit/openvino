// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

class ShuffleChannelsFusionTestValues {
public:
    bool dynamicShape;
    std::vector<int64_t> reshape_before_val;
    std::vector<size_t> transpose_val;
    std::vector<int64_t> reshape_after_val;
    size_t batch_size;
    bool check_reshape_values;
    bool fuse_happened;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class ShuffleChannelsFusion : public ::testing::Test, public testing::WithParamInterface<ShuffleChannelsFusionTestValues> {
public:
    void SetUp() override {
        const auto values = GetParam();
        {
            const PartialShape inputPartialShape = values.dynamicShape ? PartialShape::dynamic() : Shape{ values.batch_size, 128, 720, 480 };
            auto input0 = std::make_shared<opset6::Parameter>(element::f32, inputPartialShape);
            auto shape_reshape_before = opset6::Constant::create(element::i64, Shape{ values.reshape_before_val.size() }, values.reshape_before_val);
            auto permutation = opset6::Constant::create(element::i64, Shape{ values.transpose_val.size() }, values.transpose_val);
            auto shape_reshape_after = opset6::Constant::create(element::i64, Shape{ values.reshape_after_val.size() }, values.reshape_after_val);

            auto reshape_before = std::make_shared<ngraph::opset6::Reshape>(input0, shape_reshape_before, false);
            auto permute = std::make_shared<ngraph::opset6::Transpose>(reshape_before, permutation);
            auto reshape_after = std::make_shared<ngraph::opset6::Reshape>(permute, shape_reshape_after, false);
            f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input0 });

            ngraph::pass::Manager manager;
            auto pass_config = manager.get_pass_config();
            manager.register_pass<ngraph::pass::InitNodeInfo>();
            manager.register_pass<ngraph::pass::ShuffleChannelsFusion>(values.check_reshape_values);
            manager.run_passes(f);
            ASSERT_NO_THROW(check_rt_info(f));
        }

        if (values.fuse_happened) {
            auto input0 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ values.batch_size, 128, 720, 480 });
            auto shuffle_channels = std::make_shared<ngraph::opset6::ShuffleChannels>(input0, 1, values.reshape_before_val[1]);
            f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ shuffle_channels }, ngraph::ParameterVector{ input0 });
        } else {
            f_ref = f;
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsFusionTestValues> obj) {
        const ShuffleChannelsFusionTestValues testValues = obj.param;

        std::ostringstream result;
        if (testValues.dynamicShape) {
            result << "_dynamic_shape_";
        } else {
            result << "_batch_size_" << testValues.batch_size;
        }

        result << "_before_" << testValues.reshape_before_val
               << "_transpose_" << testValues.transpose_val << "_after_" << testValues.reshape_after_val
               << (testValues.check_reshape_values ? "check_reshape_values" : "");

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> f;
    std::shared_ptr<ngraph::Function> f_ref;
};

TEST_P(ShuffleChannelsFusion, CompareFunctions) {
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ShuffleChannelsFusionTestValues> testValues = {
    { true, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, 1, false, false },
    { false, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, 1, false, true },
    { false, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, 1, true, true },
    { false, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, 1, false, true },
    { false, {4, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, 4, false, false },
    { false, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, 1, true, false },
    { true, {1, 4, 32, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, 1, false, false },
    { false, {1, 4, 32, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, 1, false, true },
    { false, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, 1, true, true },
    { false, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, 1, false, true },
    { false, {4, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, 4, false, false },
    { false, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, 1, true, false },
};
INSTANTIATE_TEST_CASE_P(
    TransformationTests,
    ShuffleChannelsFusion,
    ::testing::ValuesIn(testValues),
    ShuffleChannelsFusion::getTestCaseName);
} // namespace
