// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/test_model/test_model.hpp"

#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/wstring_convert_util.hpp"

namespace ov {
namespace test {
namespace utils {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void generate_test_model(const std::wstring& model_path,
                         const std::wstring& weights_path,
                         const ov::element::Type& input_type,
                         const ov::PartialShape& input_shape) {
    ov::pass::Manager manager;
#    ifdef _WIN32
    manager.register_pass<ov::pass::Serialize>(model_path, weights_path);
#    else
    manager.register_pass<ov::pass::Serialize>(ov::util::wstring_to_string(model_path),
                                               ov::util::wstring_to_string(weights_path));
#    endif
    manager.run_passes(ov::test::utils::make_conv_pool_relu(input_shape.to_shape(), input_type));
}
#endif
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
