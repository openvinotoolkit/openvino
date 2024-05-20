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

void clip_restrict_dims(InputShape& input_shape, const std::map<size_t, size_t>& restrict_dims) {
    std::vector<ov::Shape>& staticShapes = input_shape.second;
    for (const auto& pair : restrict_dims) {
        uint64_t dimMax = std::numeric_limits<char>::max();
        uint64_t dimMin = pair.second;
        auto& dim0 = staticShapes[0][pair.first];
        auto& dim1 = staticShapes[1][pair.first];
        auto& dim2 = staticShapes[2][pair.first];
        if (dim0 != dim2) {
            dim0 = clip(dim0, dimMin, dimMax);
            dim1 = clip(dim1, dimMin, dimMax);
            dim2 = clip(dim2, dimMin, dimMax);
        }
    }
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

InputShape generate(const std::shared_ptr<ov::op::v1::Convolution>& node,
                    size_t in_port_id) {
    std::map<size_t, size_t> restrict_dims;
    for (size_t port = 0; port < node->get_input_size(); ++port) {
        if (port != in_port_id) {
            auto kernel_ptr = node->get_input_node_shared_ptr(port);
            // Layout of kernel is [C_OUT, C_IN, Z, Y, X], layout of input is [N, C_IN, Z, Y, X]
            for (size_t dim = kernel_ptr->get_shape().size() - 1; dim >= 2; --dim) {
                restrict_dims.insert({dim, kernel_ptr->get_shape()[dim]});
            }
            break;
        }
    }

    InputShape input_shape = generate(std::dynamic_pointer_cast<ov::Node>(node), in_port_id);
    clip_restrict_dims(input_shape, restrict_dims);
    return input_shape;
}

InputShape generate(const std::shared_ptr<ov::op::v1::ConvolutionBackpropData>& node,
                    size_t in_port_id) {
    std::map<size_t, size_t> restrict_dims;
    for (size_t port = 0; port < node->get_input_size(); ++port) {
        if (port != in_port_id) {
            auto kernel_ptr = node->get_input_node_shared_ptr(port);
            // Layout of kernel is [C_OUT, C_IN, Z, Y, X], layout of input is [N, C_IN, Z, Y, X]
            for (size_t dim = kernel_ptr->get_shape().size() - 1; dim >= 2; --dim) {
                restrict_dims.insert({dim, kernel_ptr->get_shape()[dim]});
            }
            break;
        }
    }

    InputShape input_shape = generate(std::dynamic_pointer_cast<ov::Node>(node), in_port_id);
    clip_restrict_dims(input_shape, restrict_dims);
    return input_shape;
}

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
