// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace test {
namespace utils {

/**
 * @brief generates IR files (XML and BIN files) with the test model.
 *        Passed reference vector is filled with OpenVINO operations to validate after the network reading.
 * @param model_path used to serialize the generated network
 * @param weights_path used to serialize the generated weights
 * @param input_type input element type of the generated model
 * @param input_shape dims on the input layer of the generated model
 */
void generate_test_model(const std::string& model_path,
                         const std::string& weights_path,
                         const ov::element::Type& input_type = ov::element::f32,
                         const ov::PartialShape& input_shape = {1, 3, 227, 227});

}  // namespace utils
}  // namespace test
}  // namespace ov
