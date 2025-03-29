// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include "plugin/transformations/convert_pooling_to_reduce.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                                const ov::element::Type& input_type,
                                                const ov::Strides& strides,
                                                const ov::Shape& pads_begin,
                                                const ov::Shape& pads_end,
                                                const ov::Shape& kernel,
                                                const bool exclude_pad,
                                                const ov::op::RoundingType rounding_type,
                                                const ov::op::PadType pad_type) {
    const auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
    const auto avgPool = std::make_shared<ov::op::v1::AvgPool>(in,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            kernel,
                                                            exclude_pad,
                                                            rounding_type,
                                                            pad_type);
    return std::make_shared<ov::Model>(ov::NodeVector{avgPool}, ov::ParameterVector{in});
}

TEST(TransformationTests, ConvertAvgPoolToReduce) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertAvgPoolingToReduce>();

    auto func = CreateFunction(
        {1, 3, 10, 10}, ov::element::Type_t::f16,
        {1, 1}, {0, 0}, {0, 0}, {10, 10},  // stride, pads_begin, pads_end, kernel
        false, ov::op::RoundingType::FLOOR, ov::op::PadType::VALID);

    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("ReduceMean") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success);
}
