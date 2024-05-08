// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/generate_static_shapes.hpp"
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace utils {

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

ov::Shape generate_mid_shape(const ov::PartialShape& partial_shape) {
    ov::Shape midShape;
    for (const auto s : partial_shape) {
        int dimValue = 1;
        if (s.is_dynamic()) {
            size_t range = s.get_max_length() - s.get_min_length();
            if (range > std::numeric_limits<char>::max()) {
                ov::test::utils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), s.get_min_length(), 1);
            }
            ov::test::utils::fill_data_random(&dimValue, 1, range, s.get_min_length(), 1);
        } else {
            dimValue = s.get_length();
        }
        midShape.push_back(dimValue);
    }
    return midShape;
}

namespace {

InputShape generate(const std::shared_ptr<ov::Node>& node,
                    size_t in_port_id) {
    const auto& param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node->get_input_node_shared_ptr(in_port_id));
    std::vector<ov::Shape> staticShapes = { param->get_partial_shape().get_min_shape(),
                                            generate_mid_shape(param->get_partial_shape()),
                                            param->get_partial_shape().get_max_shape() };
    // Shape validation to avoid large values
    uint64_t dimMin = 1;
    uint64_t dimMax = std::numeric_limits<char>::max();
    for (int i = 0; i < staticShapes[0].size(); ++i) {
        auto& dim0 = staticShapes[0][i];
        auto& dim2 = staticShapes[2][i];
        if (dim0 != dim2) {
            dim0 = clip(dim0, dimMin, dimMax);
            dim2 = clip(dim2, dimMin, dimMax);
        }
    }
    return InputShape{param->get_partial_shape(), staticShapes};
}

// namespace Reshape {
//     std::unordered_map<size_t, size_t> gen_prime_number(size_t n) {
//         // {prime_number, cnt}
//         std::unordered_map<size_t, size_t> prime_numbers = {{ 1, 1 }};
//         for (size_t i = 2; i < n / 2; ++i) {
//             while (n % i == 0) {
//                 if (prime_numbers.count(i)) {
//                     prime_numbers[i]++;
//                 } else {
//                     prime_numbers.insert({i, 1});
//                 }
//                 n /= i;
//             }
//         }
//         return prime_numbers;
//     }
// } // namespace Reshape

// InputShape generate(const std::shared_ptr<ov::op::v1::Reshape>& node,
//                     size_t in_port_id) {
//     if (in_port_id) {
//         throw std::runtime_error("SMTH");
//     }
//     std::vector<ov::Shape> static_shapes;
//     auto new_shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
//     if (new_shape_const) {
//         ov::Shape new_shape = new_shape_const->get_vector<size_t>();
//         auto shape_size = ov::shape_size(new_shape);
//         auto prime_numbers = Reshape::gen_prime_number(shape_size);
//         auto b = 0;
//     } else {
//         auto new_node = std::dynamic_pointer_cast<ov::Node>(node);
//         return generate(new_node, in_port_id);
//     }
// }

template<typename T>
InputShape generateShape(const std::shared_ptr<ov::Node>& node,
                         size_t in_port_id) {
    return generate(ov::as_type_ptr<T>(node), in_port_id);
}
} // namespace

ShapesMap getShapeMap() {
    static ShapesMap shapesMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateShape<NAMESPACE::NAME>},

#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
#include "openvino/opsets/opset13_tbl.hpp"
#include "openvino/opsets/opset14_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return shapesMap;
}
} // namespace utils
} // namespace test
} // namespace ov