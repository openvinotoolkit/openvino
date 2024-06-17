// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/tensor_parallel.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

static std::shared_ptr<ov::Model> BuildFunction() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
    auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 16 }, { 1 });
    auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input, weights_const, no_bias);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(fc, 1);

    return std::make_shared<ov::Model>(ov::NodeVector{softmax}, ov::ParameterVector{input});
}

TEST(TransformationTests, ConvertFCToTP) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::TensorParallelFusion>();
    auto model = BuildFunction();
    manager.run_passes(model);
    ov::serialize(model, "model_folder.xml");
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
