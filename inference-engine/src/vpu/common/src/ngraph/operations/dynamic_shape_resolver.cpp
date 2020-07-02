// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph/opsets/opset3.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo DynamicShapeResolver::type_info;

DynamicShapeResolver::DynamicShapeResolver(
        const Output<Node>& tensorWithData,
        const Output<Node>& tensorWithDims,
        const DynamicShapeResolverMode& mode)
    : Op(OutputVector{tensorWithData, tensorWithDims}), m_mode(mode) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> DynamicShapeResolver::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicShapeResolver>(new_args.at(0), new_args.at(1), m_mode);
}

void DynamicShapeResolver::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2, "(", get_friendly_name(), ") supports only ", 2, " inputs, but ", get_input_size(), " provided");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(1).is_static(), "(", get_friendly_name(), ") does not support dynamic shape for dims tensor");

    const auto& dataElementType = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this, dataElementType.is_static(), "(", get_friendly_name(), ") does not support dynamic element type for data tensor");
    const auto& dimsElementType = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this, dimsElementType.is_static() && (dimsElementType.compatible(ngraph::element::i64) ||
                                                                dimsElementType.compatible(ngraph::element::i32)),
        "(", get_friendly_name(), ") supports only i64 and i32 number type for dims tensor, but ", dimsElementType, " provided");

    const auto& dimsShape = get_input_shape(1);

    if (m_mode == DynamicShapeResolverMode::INFER_UPPER_BOUND_SHAPE) {
        NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static(), "(", get_friendly_name(), ") does not support dynamic shape for data tensor");

        const auto& dataShape = get_input_shape(0);
        NODE_VALIDATION_CHECK(this, dimsShape.size() == 1 && dimsShape.front() == dataShape.size(), "(", get_friendly_name(), ") inputs shapes mismatch: first "
            "input shape = ", dataShape, " second input shape = ", dimsShape, " but ", dataShape, " and ", Shape{dataShape.size()}, " are expected");

        set_output_type(0, dataElementType, dataShape);
    } else if (m_mode == DynamicShapeResolverMode::INFER_DYNAMIC_SHAPE) {
        NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).rank() == dimsShape.front(),
                "(", get_friendly_name(), ") data and shape ranks must be equal, provided: ",
                get_input_partial_shape(0).rank(), " vs ", dimsShape.front());

        set_output_type(0, dataElementType,
                        ngraph::PartialShape::dynamic(get_input_partial_shape(0).rank()));
    } else {
        NGRAPH_UNREACHABLE(this, "Unknown DynamicShapeResolverMode value, expected one of: INFER_UPPER_BOUND_SHAPE, INFER_DYNAMIC_SHAPE");
    }
}

bool DynamicShapeResolver::visit_attributes(ngraph::AttributeVisitor&) {
    return true;
}

namespace {

template<element::Type_t ET>
bool getShapeFromHostTensorData(const HostTensorPtr& data, Shape& result) {
    using T = typename element_type_traits<ET>::value_type;
    T *dataPtr = data->get_data_ptr<ET>();
    if (!dataPtr) {
        return false;
    }
    if (data->get_shape().size() != 1) {
        return false;
    }
    size_t outputRank = data->get_shape()[0];

    for (int i = 0; i < outputRank; i++) {
        result.push_back(dataPtr[i]);
    }

    return true;
}

bool getShapeFromHostTensorData(const HostTensorPtr& data, Shape& shape) {
    switch (data->get_element_type()) {
        case element::Type_t::i8:
            return getShapeFromHostTensorData<element::Type_t::i8>(data, shape);
        case element::Type_t::i16:
            return getShapeFromHostTensorData<element::Type_t::i16>(data, shape);
        case element::Type_t::i32:
            return getShapeFromHostTensorData<element::Type_t::i32>(data, shape);
        case element::Type_t::i64:
            return getShapeFromHostTensorData<element::Type_t::i64>(data, shape);
        case element::Type_t::u8:
            return getShapeFromHostTensorData<element::Type_t::u8>(data, shape);
        case element::Type_t::u16:
            return getShapeFromHostTensorData<element::Type_t::u16>(data, shape);
        case element::Type_t::u32:
            return getShapeFromHostTensorData<element::Type_t::u32>(data, shape);
        case element::Type_t::u64:
            return getShapeFromHostTensorData<element::Type_t::u64>(data, shape);
        default:
            return false;
    }
    return true;
}

template<element::Type_t DataType>
bool evaluate(const HostTensorPtr& inputTensor,
              const HostTensorPtr& inputShapeTensor,
              const HostTensorPtr& outputTensor) {
    Shape inputShape = inputTensor->get_shape();
    Shape outputShape;
    if (!getShapeFromHostTensorData(inputShapeTensor, outputShape)) {
        return false;
    }

    if (!ngraph::PartialShape(outputShape).refines(outputTensor->get_partial_shape())) {
        return false;
    }

    outputTensor->set_shape(outputShape);

    using T = typename element_type_traits<DataType>::value_type;
    T *inputPtr = inputTensor->get_data_ptr<DataType>();
    T *outputPtr = outputTensor->get_data_ptr<DataType>();

    const auto inTotalDimSize = shape_size(inputShape);
    const auto stridesByElements = row_major_strides(inputShape);

    const auto inLineSize = inputShape[inputShape.size() - 1];
    const auto outLineSize = outputShape[outputShape.size() - 1];

    for (size_t inElementOffset = 0, outElementOffset = 0; inElementOffset < inTotalDimSize; inElementOffset += inLineSize) {
        auto offset = inElementOffset;
        bool isGarbageLine = false;
        for (size_t dim = 0; dim < stridesByElements.size() - 1; ++dim) {
            const auto coordAlongDim = offset / stridesByElements[dim];
            if (coordAlongDim > outputShape[dim] - 1) {
                isGarbageLine = true;
                break;
            }

            offset %= stridesByElements[dim];
        }
        if (!isGarbageLine) {
            std::copy_n(inputPtr + inElementOffset, outLineSize, outputPtr + outElementOffset);
            outElementOffset += outLineSize;
        }
    }
    return true;
}

bool evaluateDynamicShapeResolver(const HostTensorPtr& inputTensor,
                                  const HostTensorPtr& inputShapeTensor,
                                  const HostTensorPtr& outputTensor) {
    bool rc = true;

    switch (inputTensor->get_element_type()) {
        TYPE_CASE(i8)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(i16)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(i32)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(i64)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(u8)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(u16)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(u32)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(u64)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(bf16)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(f32)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(f64)(inputTensor, inputShapeTensor, outputTensor);
            break;
        TYPE_CASE(boolean)(inputTensor, inputShapeTensor, outputTensor);
            break;
        default:
            rc = false;
            break;
    }

    return rc;
}

}  // namespace

bool DynamicShapeResolver::evaluate(const HostTensorVector& outputs,
                                    const HostTensorVector& inputs) {
    return evaluateDynamicShapeResolver(inputs[0], inputs[1], outputs[0]);
}

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
