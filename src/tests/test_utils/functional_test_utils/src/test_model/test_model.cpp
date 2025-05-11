// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/test_model/test_model.hpp"

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

namespace ov {
namespace test {
namespace utils {

void generate_test_model(const std::string& model_path,
                         const std::string& weights_path,
                         const ov::element::Type& input_type,
                         const ov::PartialShape& input_shape) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(model_path, weights_path);
    manager.run_passes(ov::test::utils::make_conv_pool_relu(input_shape.to_shape(), input_type));
}

}  // namespace utils
}  // namespace test
}  // namespace ov
