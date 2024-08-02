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

namespace pooling {
inline size_t dilated(size_t dim, size_t dilation) {
    return (dim - 1) * dilation + 1;
}

int64_t get_min_spatial_dim(size_t padBegin, size_t padEnd, size_t kernel, size_t dilation) {
    return dilated(kernel, dilation) - static_cast<int64_t>(padBegin + padEnd);
}

InputShape generatePoolingShape(const ov::PartialShape& partialShape, const Strides& dilations, const Shape& kernel,
                               const Shape& padsBegin, const Shape& padsEnd) {
    std::vector<ov::Shape> staticShapes = { partialShape.get_min_shape(),
                                            ov::Shape(),
                                            partialShape.get_max_shape() };

    auto clip_spatial_dim = [&](Shape& shape) {
        for (size_t i = 2; i < shape.size(); ++i) {
            shape[i] = std::max(shape[i], static_cast<size_t>(get_min_spatial_dim(padsBegin[i - 2], padsEnd[i - 2], kernel[i - 2], dilations[i - 2])));
        }
    };

    clip_spatial_dim(staticShapes[0]);
    clip_spatial_dim(staticShapes[2]);

    for (size_t i = 0; i < partialShape.size(); ++i) {
        const auto& s = partialShape[i];
        int dimValue = 1;
        if (s.is_dynamic()) {
            size_t range = s.get_max_length() - s.get_min_length();
            if (range > std::numeric_limits<char>::max()) {
                ov::test::utils::fill_data_random(&range, 1, std::numeric_limits<char>::max(), s.get_min_length(), 1);
            }
            const auto minValue = i > 1 ? get_min_spatial_dim(padsBegin[i - 2], padsEnd[i - 2], kernel[i - 2], dilations[i - 2]) : 1;
            ov::test::utils::fill_data_random(&dimValue, 1, range, std::max(s.get_min_length(), minValue), 1);
        } else {
            dimValue = s.get_length();
        }

        staticShapes[1].push_back(dimValue);
    }

    // Shape validation to avoid large values
    uint64_t dimMin = 1;
    uint64_t dimMax = std::numeric_limits<char>::max();
    for (auto& shape : staticShapes) {
        for (auto& dimValue : shape)
            dimValue = clip(dimValue, dimMin, dimMax);
    }
    return InputShape{partialShape, staticShapes};
}
} // namespace pooling

InputShape generate(const std::shared_ptr<ov::op::v1::MaxPool>& node,
                   size_t port) {
    const auto& partialShape = node->get_input_partial_shape(port);
    const auto& padsBegin = node->get_pads_begin();
    const auto& padsEnd = node->get_pads_end();
    const auto& kernel = node->get_kernel();
    const auto dilations = ov::Strides(kernel.size(), 1);
    return pooling::generatePoolingShape(partialShape, dilations, kernel, padsBegin, padsEnd);
}

InputShape generate(const std::shared_ptr<ov::op::v8::MaxPool>& node,
                   size_t port) {
    const auto& partialShape = node->get_input_partial_shape(port);
    const auto& padsBegin = node->get_pads_begin();
    const auto& padsEnd = node->get_pads_end();
    const auto& kernel = node->get_kernel();
    const auto& dilations =  node->get_dilations();
    return pooling::generatePoolingShape(partialShape, dilations, kernel, padsBegin, padsEnd);
}

InputShape generate(const std::shared_ptr<ov::op::v14::MaxPool>& node,
                   size_t port) {
    const auto& partialShape = node->get_input_partial_shape(port);
    const auto& padsBegin = node->get_pads_begin();
    const auto& padsEnd = node->get_pads_end();
    const auto& kernel = node->get_kernel();
    const auto& dilations =  node->get_dilations();
    return pooling::generatePoolingShape(partialShape, dilations, kernel, padsBegin, padsEnd);
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
