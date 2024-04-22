// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <math.h>
#include <algorithm>
#include <functional>

#include "shared_test_classes/base/utils/generate_static_shapes.hpp"

#include "openvino/op/ops.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/roi_align.hpp"

#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace utils {

uint64_t clip(uint64_t n, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(n, upper));
}

namespace {
InputShape generate(const std::shared_ptr<ov::op::v1::Convolution>& node,
                    const std::shared_ptr<ov::op::v0::Parameter>& param) {
    std::cout << "in generate convolution" << std::endl;
    std::map<size_t, size_t>restrict_shape;
    for (size_t i = 0; i < param->get_output_size(); i++) {
        for (const auto &node : param->get_output_target_inputs(i)) {
            std::shared_ptr<ov::Node> nodePtr1 = node.get_node()->shared_from_this();
            std::cout << "inputsize: " << nodePtr1->get_input_size() << std::endl;
            for (size_t port = 0; port < nodePtr1->get_input_size(); ++port) {
                if (nodePtr1->get_input_node_ptr(port)->shared_from_this() != param) {
                    std::shared_ptr<ov::op::v0::Constant> constant_ptr =
                        std::dynamic_pointer_cast<ov::op::v0::Constant>(nodePtr1->get_input_node_shared_ptr(port));
                    std::cout << "!=port: " << port << " constant shape: " << constant_ptr->get_shape() << std::endl;
                    restrict_shape.insert({2, *constant_ptr->get_shape().end()});
                    break;
                }
            }
        }
    }

    const std::shared_ptr<ov::Node> nodecast_ptr = std::dynamic_pointer_cast<ov::Node>(node);
    InputShape input_shape = generate(nodecast_ptr, param);
    auto staticShapes = input_shape.second;
    for (const auto& pair : restrict_shape) {
        uint64_t dimMax = std::numeric_limits<char>::max(); // 127
        uint64_t dimMin = pair.second;
        size_t i = pair.first;
        auto& dim0 = staticShapes[0][i];
        auto& dim2 = staticShapes[2][i];
        if (dim0 != dim2) {
            dim0 = clip(dim0, dimMin, dimMax);
            dim2 = clip(dim2, dimMin, dimMax);
        }
    }
    return input_shape;
    //return InputShape{{}, {}};
}


InputShape generate(const std::shared_ptr<ov::Node>& node,
                    const std::shared_ptr<ov::op::v0::Parameter>& param) {
    std::cout << "in generate" << std::endl;
    // [0,512,1]
    // [9223372036854775807,512,9223372036854775807]
    std::vector<ov::Shape> staticShapes = { param->get_partial_shape().get_min_shape(),
                                            param->get_partial_shape().get_min_shape(),
                                            param->get_partial_shape().get_max_shape() };
    ov::Shape midShape;
    for (const auto s : param->get_partial_shape()) {
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
    staticShapes[1] = midShape;

    // Shape validation to avoid large values
    uint64_t dimMin = 1;
    uint64_t dimMax = std::numeric_limits<char>::max(); // 127
    std::cout << "dim min: " << dimMin << " dim max: " << dimMax << std::endl;
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

template<typename T>
InputShape generateShape(const std::shared_ptr<ov::Node>& node,
                         const std::shared_ptr<ov::op::v0::Parameter> &param) {
    return generate(ov::as_type_ptr<T>(node), param);
}
} // namespace

ShapesMap getShapeMap() {
    static ShapesMap shapesMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateShape<NAMESPACE::NAME>},

#include "openvino/opsets/opset1_tbl.hpp"
//#include "openvino/opsets/opset2_tbl.hpp"
//#include "openvino/opsets/opset3_tbl.hpp"
//#include "openvino/opsets/opset4_tbl.hpp"
//#include "openvino/opsets/opset5_tbl.hpp"
//#include "openvino/opsets/opset6_tbl.hpp"
//#include "openvino/opsets/opset7_tbl.hpp"
//#include "openvino/opsets/opset8_tbl.hpp"
//#include "openvino/opsets/opset9_tbl.hpp"
//#include "openvino/opsets/opset10_tbl.hpp"
//#include "openvino/opsets/opset11_tbl.hpp"
//#include "openvino/opsets/opset12_tbl.hpp"
//#include "openvino/opsets/opset13_tbl.hpp"

//#include "ov_ops/opset_private_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return shapesMap;
}
} // namespace utils
} // namespace test
} // namespace ov