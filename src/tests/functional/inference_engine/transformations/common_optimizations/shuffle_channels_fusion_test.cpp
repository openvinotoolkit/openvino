// Copyright (C) 2018-2022 Intel Corporation
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
#include "common_test_utils/common_utils.hpp"

namespace {
using namespace testing;
using namespace ngraph;

enum FuseHappened {
    YES,
    NO
};

class ShuffleChannelsFusionTestValues {
public:
    ngraph::PartialShape inputPartialShape;
    std::vector<int64_t> reshape_before_val;
    std::vector<size_t> transpose_val;
    std::vector<int64_t> reshape_after_val;
    bool check_values;
    FuseHappened fuse_happened;
};

class ShuffleChannelsFusion : public ::testing::Test, public testing::WithParamInterface<ShuffleChannelsFusionTestValues> {
public:
    void SetUp() override {
        const auto values = GetParam();
        {
            auto input0 = std::make_shared<opset6::Parameter>(element::f32, values.inputPartialShape);
            auto shape_reshape_before = opset6::Constant::create(element::i64, Shape{ values.reshape_before_val.size() }, values.reshape_before_val);
            auto permutation = opset6::Constant::create(element::i64, Shape{ values.transpose_val.size() }, values.transpose_val);
            auto shape_reshape_after = opset6::Constant::create(element::i64, Shape{ values.reshape_after_val.size() }, values.reshape_after_val);

            auto reshape_before = std::make_shared<ngraph::opset6::Reshape>(input0, shape_reshape_before, true);
            auto permute = std::make_shared<ngraph::opset6::Transpose>(reshape_before, permutation);
            auto reshape_after = std::make_shared<ngraph::opset6::Reshape>(permute, shape_reshape_after, true);
            f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input0 });
        }

        if (values.fuse_happened == FuseHappened::YES) {
            auto input0 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, values.inputPartialShape);
            auto shuffle_channels = std::make_shared<ngraph::opset6::ShuffleChannels>(input0, 1, values.reshape_before_val[1]);
            f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ shuffle_channels }, ngraph::ParameterVector{ input0 });
        } else {
            f_ref = f;
        }

        auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
        ngraph::pass::Manager manager;
        auto pass_config = manager.get_pass_config();
        manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ShuffleChannelsFusion>(values.check_values);
        manager.register_pass<ngraph::pass::CheckUniqueNames>(unh);
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    static std::string getTestCaseName(testing::TestParamInfo<ShuffleChannelsFusionTestValues> obj) {
        const ShuffleChannelsFusionTestValues testValues = obj.param;

        std::ostringstream result;
        result << "_input_shape_" << testValues.inputPartialShape
               << "_before_" << CommonTestUtils::vec2str(testValues.reshape_before_val)
               << "_transpose_" << CommonTestUtils::vec2str(testValues.transpose_val)
               << "_after_" << CommonTestUtils::vec2str(testValues.reshape_after_val)
               << (testValues.check_values ? "check_reshape_values" : "");

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> f;
    std::shared_ptr<ngraph::Function> f_ref;
};

TEST_P(ShuffleChannelsFusion, CompareFunctions) {
    auto fc = FunctionsComparator::no_default()
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::NODES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

const std::vector<ShuffleChannelsFusionTestValues> testValues = {
    // dynamic shape
    { PartialShape::dynamic(), {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, true, FuseHappened::NO},
    { PartialShape::dynamic(), {1, 4, 32, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, false, FuseHappened::NO },

    // 4D, dynamic batch
    { PartialShape{Dimension::dynamic(), 128, 720, 480}, {0, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {0, 128, 720, 480}, true, FuseHappened::YES },
    { PartialShape{Dimension::dynamic(), 128, 720, 480}, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, true, FuseHappened::YES },
    { PartialShape{Dimension::dynamic(), 128, 720, 480}, {4, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {4, 128, 720, 480}, true, FuseHappened::YES },
    { PartialShape{Dimension::dynamic(), 128, 720, 480}, {0, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {0, -1, 720, 480}, false, FuseHappened::YES },
    { PartialShape{Dimension::dynamic(), 128, 720, 480}, {4, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {4, -1, 720, 480}, false, FuseHappened::YES },

    // 4D, batch_size = 1, 4D reshape constant
    { {1, 128, 720, 480}, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, false, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, 128, 720, 480}, true, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, false, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, true, FuseHappened::NO },

    // 4D, batch_size = 1, 3D reshape constant
    { {1, 128, 720, 480}, {1, 4, 32, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, false, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, 128, 720, 480}, true, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, false, FuseHappened::YES },
    { {1, 128, 720, 480}, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, true, FuseHappened::NO },

    // 4D, batch_size = 4
    { {4, 128, 720, 480}, {4, 2, 64, 720, 480}, {0, 2, 1, 3, 4},  {1, -1, 720, 480}, false, FuseHappened::NO },
    { {4, 128, 720, 480}, {4, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, false, FuseHappened::NO },

    // 2D
    { {128, 720 * 480}, {1, 2, 64, 720 * 480}, {0, 2, 1, 3},  {1, -1, 720, 480}, false, FuseHappened::NO },
};

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    ShuffleChannelsFusion,
    ::testing::ValuesIn(testValues),
    ShuffleChannelsFusion::getTestCaseName);
} // namespace
