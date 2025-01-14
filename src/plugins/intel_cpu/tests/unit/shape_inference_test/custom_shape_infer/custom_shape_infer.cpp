// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom_shape_infer.hpp"

#include <gtest/gtest.h>

#include "openvino/cc/factory.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "shape_inference/custom/adaptive_pooling.hpp"
#include "shape_inference/custom/color_convert.hpp"
#include "shape_inference/custom/eltwise.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "shape_inference/custom/gather.hpp"
#include "shape_inference/custom/matmul.hpp"
#include "shape_inference/custom/ngram.hpp"
#include "shape_inference/custom/one_hot.hpp"
#include "shape_inference/custom/priorbox.hpp"
#include "shape_inference/custom/priorbox_clustered.hpp"
#include "shape_inference/custom/reshape.hpp"
#include "shape_inference/custom/scaled_attn.hpp"
#include "shape_inference/custom/shapeof.hpp"
#include "shape_inference/custom/strided_slice.hpp"
#include "shape_inference/custom/transpose.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace {
#define INTEL_CPU_CUSTOM_SHAPE_INFER(__prim, __type) \
    registerNodeIfRequired(intel_cpu, __prim, __type, __prim)

class EltwiseShapeInferTestFactory : public node::EltwiseShapeInferFactory {
public:
    EltwiseShapeInferTestFactory(std::shared_ptr<ov::Node> op) : EltwiseShapeInferFactory() {}
};

class ShapeOfShapeInferTestFactory : public node::ShapeOfShapeInferFactory {
public:
    ShapeOfShapeInferTestFactory(std::shared_ptr<ov::Node> op) : ShapeOfShapeInferFactory() {}
};

class CustomShapeInferFF : public openvino::cc::Factory<Type, ShapeInferFactory*(const std::shared_ptr<ov::Node>& op)> {
public:
    CustomShapeInferFF():Factory("CpuCustomShapeInferTestFactory") {
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::AdaptivePoolingShapeInferFactory, Type::AdaptivePooling);
    INTEL_CPU_CUSTOM_SHAPE_INFER(EltwiseShapeInferTestFactory, Type::Eltwise);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::FCShapeInferFactory, Type::FullyConnected);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::TransposeShapeInferFactory, Type::Transpose);
    INTEL_CPU_CUSTOM_SHAPE_INFER(ShapeOfShapeInferTestFactory, Type::ShapeOf);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::ColorConvertShapeInferFactory, Type::ColorConvert);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::ReshapeShapeInferFactory, Type::Reshape);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::MMShapeInferFactory, Type::MatMul);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::OneHotShapeInferFactory, Type::OneHot);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::StridedSliceShapeInferFactory, Type::StridedSlice);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::PriorBoxShapeInferFactory, Type::PriorBox);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::PriorBoxClusteredShapeInferFactory, Type::PriorBoxClustered);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::NgramShapeInferFactory, Type::Ngram);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::GatherShapeInferFactory, Type::Gather);
    INTEL_CPU_CUSTOM_SHAPE_INFER(node::SDPAShapeInferFactory, Type::ScaledDotProductAttention);
#undef INTEL_CPU_CUSTOM_SHAPE_INFER
    }

    ShapeInferFactory* create(const std::shared_ptr<ov::Node>& op) {
        ShapeInferFactory* newShapeInferFactory = nullptr;
        std::unique_ptr<ShapeInferFactory> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op));
        if (ol != nullptr) {
             newShapeInferFactory = ol.release();
        }
        return newShapeInferFactory;
    }
};

void compare_result(const std::vector<StaticShape>& ref, const std::vector<VectorDims>& cus) {
    ASSERT_EQ(ref.size(), cus.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_EQ(ref[i].size(), cus[i].size());
        for (size_t y = 0; y < ref[i].size(); y++) {
            ASSERT_EQ(ref[i][y].get_length(), cus[i][y]);
        }
    }
}

} //namespace

void cpu_test_shape_infer(ov::Node* op,
                          const std::vector<StaticShape>& input_shapes,
                          std::vector<StaticShape>& output_shapes,
                          const std::unordered_map<size_t, ov::Tensor>& constant_data) {
    static std::shared_ptr<CustomShapeInferFF> cusFactory = std::make_shared<CustomShapeInferFF>();
    auto shapeInferFactory = cusFactory->create(op->shared_from_this());
    ASSERT_TRUE(shapeInferFactory != nullptr);
    auto cusShapeInfer =  shapeInferFactory->makeShapeInfer();
    std::vector<std::reference_wrapper<const VectorDims>> cusInputShapes;
    std::vector<VectorDims> tmpInputShapes;
    cusInputShapes.reserve(input_shapes.size());
    tmpInputShapes.reserve(input_shapes.size());
    for (size_t port = 0; port < input_shapes.size(); ++port) {
        VectorDims dims;
        for (size_t i = 0; i < input_shapes[port].size(); ++i) {
            dims.emplace_back(input_shapes[port][i].get_length());
        }
        tmpInputShapes.emplace_back(dims);
        cusInputShapes.emplace_back(std::ref(tmpInputShapes[port]));
    }

    std::unordered_map<size_t, MemoryPtr> cusInputValues;
    auto input_value_port_mask = cusShapeInfer->get_port_mask();
    dnnl::engine eng;
    if (input_value_port_mask) {
        for (size_t port = 0; port < input_shapes.size(); ++port) {
            if (input_value_port_mask & (1 << port)) {
                const auto tensorIter = constant_data.find(port);
                const void* data = nullptr;
                ov::element::Type elementType;
                if (tensorIter != constant_data.end()) {
                    const auto& tensor = tensorIter->second;
                    data = tensor.data();
                    elementType = tensor.get_element_type();
                } else {
                    const auto input_op = op->input_value(port).get_node_shared_ptr();
                    const auto const_op = ov::as_type_ptr<const ov::op::v0::Constant>(input_op);
                    ASSERT_TRUE(const_op != nullptr);
                    data = const_op->get_data_ptr();
                    elementType = const_op->get_element_type();
                }
                CpuBlockedMemoryDesc desc(
                        elementType,
                        ov::intel_cpu::Shape(tmpInputShapes[port]));
                MemoryPtr memoryPtr = std::make_shared<Memory>(eng, desc, data, true);
                cusInputValues[port] = memoryPtr;
            }
        }
    }
    auto result = cusShapeInfer->infer(cusInputShapes, cusInputValues);
    compare_result(output_shapes, result.dims);
    ASSERT_TRUE(result.status == ov::intel_cpu::ShapeInferStatus::success);
}

std::string boolToString(const bool value) {
    return value ? "true" : "false";
}

} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
