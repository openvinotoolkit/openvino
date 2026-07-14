// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

using DynamicQuantize = ov::op::internal::DynamicQuantize;

static DynamicQuantize::Attributes make_dq_attrs(uint64_t group_size, size_t input_rank = 3) {
    DynamicQuantize::Attributes attrs;
    attrs.quantization_type = DynamicQuantize::QuantizationType::Symmetric;
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.zp_dt = ov::element::dynamic;
    attrs.output_storage_type = DynamicQuantize::OutputStorageType::Planar;
    attrs.group_sizes = std::vector<uint64_t>(input_rank, 1);
    attrs.group_sizes.back() = group_size;
    return attrs;
}

// 2 identical DQ nodes sharing same input merged into 1
TEST_F(TransformationTestsF, DynamicQuantizeSharing_TwoConsumers) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto dq_attrs = make_dq_attrs(64);
        auto dq1 = std::make_shared<DynamicQuantize>(input, dq_attrs);
        auto dq2 = std::make_shared<DynamicQuantize>(input, dq_attrs);

        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq1->output(0), weight1, bias, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq2->output(0), weight2, bias, scale2);

        auto result1 = std::make_shared<ov::op::v0::Result>(fc1);
        auto result2 = std::make_shared<ov::op::v0::Result>(fc2);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto dq_attrs = make_dq_attrs(64);
        auto dq_shared = std::make_shared<DynamicQuantize>(input, dq_attrs);

        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq_shared->output(0), weight1, bias, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq_shared->output(0), weight2, bias, scale2);

        auto result1 = std::make_shared<ov::op::v0::Result>(fc1);
        auto result2 = std::make_shared<ov::op::v0::Result>(fc2);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{input});
    }
}

// 5 identical DQ nodes sharing same input merged into 1
TEST_F(TransformationTestsF, DynamicQuantizeSharing_FiveConsumers) {
    const size_t num_consumers = 5;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto dq_attrs = make_dq_attrs(64);

        ov::ResultVector results;
        for (size_t i = 0; i < num_consumers; i++) {
            auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
            auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
            auto dq = std::make_shared<DynamicQuantize>(input, dq_attrs);
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq->output(0), weight, bias, scale);
            results.push_back(std::make_shared<ov::op::v0::Result>(fc));
        }
        model = std::make_shared<ov::Model>(results, ov::ParameterVector{input});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto dq_attrs = make_dq_attrs(64);
        auto dq_shared = std::make_shared<DynamicQuantize>(input, dq_attrs);

        ov::ResultVector results;
        for (size_t i = 0; i < num_consumers; i++) {
            auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
            auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
            auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq_shared->output(0), weight, bias, scale);
            results.push_back(std::make_shared<ov::op::v0::Result>(fc));
        }
        model_ref = std::make_shared<ov::Model>(results, ov::ParameterVector{input});
    }
}

// DQ nodes with different group sizes should NOT be merged
TEST_F(TransformationTestsF, DynamicQuantizeSharing_DifferentGroupSize_NoMerge) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto dq1 = std::make_shared<DynamicQuantize>(input, make_dq_attrs(64));
        auto dq2 = std::make_shared<DynamicQuantize>(input, make_dq_attrs(128));

        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq1->output(0), weight1, bias, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq2->output(0), weight2, bias, scale2);

        auto result1 = std::make_shared<ov::op::v0::Result>(fc1);
        auto result2 = std::make_shared<ov::op::v0::Result>(fc2);
        model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::SharedOpOptimization>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 4096});
        auto weight1 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto weight2 = std::make_shared<ov::op::v0::Constant>(ov::element::u4, ov::Shape{1024, 4096});
        auto scale1 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 64});
        auto scale2 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1024, 32});
        auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto dq1 = std::make_shared<DynamicQuantize>(input, make_dq_attrs(64));
        auto dq2 = std::make_shared<DynamicQuantize>(input, make_dq_attrs(128));

        auto fc1 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq1->output(0), weight1, bias, scale1);
        auto fc2 = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(dq2->output(0), weight2, bias, scale2);

        auto result1 = std::make_shared<ov::op::v0::Result>(fc1);
        auto result2 = std::make_shared<ov::op::v0::Result>(fc2);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result1, result2}, ov::ParameterVector{input});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
