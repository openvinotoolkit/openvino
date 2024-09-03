// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/remove_unnecessary_pad.hpp"

using namespace ov;
using namespace ov::intel_gpu;
using namespace ov::op;
using namespace ov::pass;

namespace ov {
namespace test {
namespace intel_gpu {

TEST(RemoveUnnecessaryPadTest, RemovePadWithAllZeroPads) {    
    auto input_shape = Shape{1, 3, 224, 224};
    auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

    auto pads_begin = v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto pads_end = v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto pad_value = v0::Constant::create(element::f32, Shape{}, {0.0f});

    auto pad = std::make_shared<v12::Pad>(input, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
    auto max_pool = std::make_shared<v1::MaxPool>(pad, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2});

    auto model = std::make_shared<ov::Model>(NodeVector{max_pool}, ParameterVector{input});

    pass::Manager manager;
    manager.register_pass<RemoveUnnecessaryPad>();
    manager.run_passes(model);

    auto new_max_pool = std::dynamic_pointer_cast<v1::MaxPool>(model->get_result()->get_input_node_shared_ptr(0));
    ASSERT_NE(new_max_pool, nullptr);
    ASSERT_NE(new_max_pool->input_value(0).get_node_shared_ptr(), pad);
    ASSERT_EQ(new_max_pool->input_value(0).get_node_shared_ptr(), input);
}

TEST(RemoveUnnecessaryPadTest, KeepPadWithNonZeroPads) {
    auto input_shape = Shape{1, 3, 224, 224};
    auto input = std::make_shared<v0::Parameter>(element::f32, input_shape);

    auto pads_begin = v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1});
    auto pads_end = v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1});
    auto pad_value = v0::Constant::create(element::f32, Shape{}, {0.0f});

    auto pad = std::make_shared<v12::Pad>(input, pads_begin, pads_end, pad_value, op::PadMode::CONSTANT);
    auto max_pool = std::make_shared<v1::MaxPool>(pad, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2});

    auto model = std::make_shared<ov::Model>(NodeVector{max_pool}, ParameterVector{input});

    pass::Manager manager;
    manager.register_pass<RemoveUnnecessaryPad>();
    manager.run_passes(model);

    auto new_max_pool = std::dynamic_pointer_cast<v1::MaxPool>(model->get_result()->get_input_node_shared_ptr(0));
    ASSERT_NE(new_max_pool, nullptr);
    ASSERT_EQ(new_max_pool->input_value(0).get_node_shared_ptr(), pad);
}
}  // namespace intel_gpu
}  // namespace test
}  // namespace ov