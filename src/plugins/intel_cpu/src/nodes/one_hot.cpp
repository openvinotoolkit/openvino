// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <dnnl_types.h>
#include "ie_parallel.hpp"
#include <selective_build.h>
#include "one_hot.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include <ie_ngraph_utils.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
/**
 * Implements One Hot shape inference algorithm. The output shape is the input `indices` tensor shape, where a new axis
 * of size `depth` is inserted at the dimension defined by the `axis` parameter.
 *  
 */
class OneHotShapeInfer : public ShapeInferEmptyPads {
public:
    explicit OneHotShapeInfer(int64_t axis) : m_axis(axis) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        auto depth = reinterpret_cast<int32_t *>(data_dependency.at(1)->GetPtr())[0];

        auto result = input_shapes.front().get();
        result.insert(result.begin() + m_axis, depth);

        return {{std::move(result)}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

private:
    int64_t m_axis = 0;
};

class OneHotShapeInferFactory : public ShapeInferFactory {
public:
    OneHotShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        auto oneHot = ov::as_type_ptr<const ngraph::opset1::OneHot>(m_op);
        if (!oneHot) {
            IE_THROW() << "Unexpected op type in OneHot shape inference factory: " << m_op->get_type_name();
        }
        auto axis = oneHot->get_axis();
        auto dstShape = oneHot->get_output_partial_shape(0);
        int output_dims_size = dstShape.size();
        if (0 == output_dims_size) output_dims_size = 1;
        if (axis < 0) {
            axis += output_dims_size;
        }
        return std::make_shared<OneHotShapeInfer>(axis);
    }

private:
    std::shared_ptr<ov::Node> m_op;
};

} // namespace

bool OneHot::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
        if (!oneHot) {
            errorMessage = "Only opset1 OneHot operation is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(ON_VALUE_ID)) == nullptr) {
            errorMessage = "Only const 'on_value' input is supported";
            return false;
        }
        if (std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(OFF_VALUEAXES_ID)) == nullptr) {
            errorMessage = "Only const 'off_value' input is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

OneHot::OneHot(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, OneHotShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "OneHot layer with name '" + op->get_friendly_name() + "'";
    const auto oneHot = std::dynamic_pointer_cast<const ngraph::opset1::OneHot>(op);
    const auto depthNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(oneHot->get_input_node_shared_ptr(DEPTH_ID));
    if (depthNode) {
        depth = depthNode->cast_vector<uint32_t>()[0];
    }
    axis = oneHot->get_axis();

    VectorDims srcDims = getInputShapeAtPort(INDICES_ID).getDims();
    if (ngraph::is_scalar(srcDims)) {
        srcDims = SizeVector{1};
    }
    VectorDims dstDims = getOutputShapeAtPort(0).getDims();
    if (ngraph::is_scalar(dstDims)) {
        dstDims = SizeVector{1};
    }

    int output_dims_size = dstDims.size();
    if (axis < 0) {
        axis += output_dims_size;
    }
    if (axis < 0 || axis >= output_dims_size) {
        IE_THROW() << errorPrefix << " has unsupported 'axis' attribute: " << oneHot->get_axis();
    }

    if (!(((1 + srcDims.size()) == dstDims.size()) ||
            (depthNode && (srcDims.size() == 1 && dstDims.size() == 1 && dstDims[0] == depth && srcDims[0] == 1))))
        IE_THROW() << errorPrefix << " has incorrect number of input/output dimensions!";
}

bool OneHot::needShapeInfer() const {
    const auto depthNodePtr = reinterpret_cast<int32_t *>(getParentEdgesAtPort(1)[0]->getMemoryPtr()->GetPtr());
    if (depth != depthNodePtr[0]) {
        depth = depthNodePtr[0];
        return true;
    }

    return Node::needShapeInfer();
}

void OneHot::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // check a precision of the input tensor
    auto input_precision = getOriginalInputPrecisionAtPort(INDICES_ID);
    if (input_precision != Precision::I32) {
        IE_THROW() << errorPrefix << " has incorrect input precision for the input. Only I32 is supported!";
    }
    output_precision = getOriginalOutputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, input_precision},
                          {LayoutType::ncsp, input_precision},
                          {LayoutType::ncsp, output_precision},
                          {LayoutType::ncsp, output_precision}},
                         {{LayoutType::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

template<typename out_type>
void OneHot::one_hot(size_t prefix_size, size_t suffix_size) {
    const auto *src_data = reinterpret_cast<const in_type *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<out_type *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    const out_type on_value = reinterpret_cast<const out_type *>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr())[0];
    const out_type off_value = reinterpret_cast<const out_type *>(getParentEdgeAt(3)->getMemoryPtr()->GetPtr())[0];

    // fill the output with off_value
    std::size_t dst_size = prefix_size * depth * suffix_size;
    std::fill(dst_data, dst_data + dst_size, off_value);

    // set on_value at needed locations
    auto on_val = on_value;
    parallel_for(prefix_size, [&](std::size_t prefix_idx) {
        const in_type* src_dataPtr = &src_data[prefix_idx * suffix_size];
        out_type* dst_dataPtr = &dst_data[prefix_idx * depth * suffix_size];
        for (std::size_t suffix_idx = 0; suffix_idx < suffix_size; ++suffix_idx, ++src_dataPtr, ++dst_dataPtr) {
            auto v = static_cast<std::size_t>(*src_dataPtr);
            if (v < depth) {
                dst_dataPtr[v * suffix_size] = on_val;
            }
        }
    });
}

void OneHot::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void OneHot::execute(dnnl::stream strm) {
    std::size_t prefix_size = 1;
    auto input_dims = getParentEdgeAt(0)->getMemory().getStaticDims();

    std::size_t actual_axis = (axis == -1) ? input_dims.size() : axis;
    for (size_t i = 0; i < actual_axis; ++i)
        prefix_size *= input_dims[i];

    std::size_t suffix_size = getParentEdgeAt(0)->getMemory().GetShape().getElementsCount() / prefix_size;

    OneHotContext ctx = {this, prefix_size, suffix_size};
    OV_SWITCH(intel_cpu, OneHotExecute, ctx, output_precision.size(),
              OV_CASE(sizeof(uint32_t), uint32_t),
              OV_CASE(sizeof(uint16_t), uint16_t),
              OV_CASE(sizeof(uint8_t), uint8_t))
}

bool OneHot::created() const {
    return getType() == Type::OneHot;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
