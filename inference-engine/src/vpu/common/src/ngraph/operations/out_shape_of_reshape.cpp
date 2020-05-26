// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/error.hpp>
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"

namespace ngraph { namespace vpu { namespace op {

constexpr NodeTypeInfo OutShapeOfReshape::type_info;

OutShapeOfReshape::OutShapeOfReshape(
        const Output<Node>& inDataShape,
        const Output<Node>& outShapeDescriptor,
        bool specialZero) : Op({inDataShape, outShapeDescriptor}), m_specialZero(specialZero) {
    constructor_validate_and_infer_types();
}

void OutShapeOfReshape::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have only 2 inputs, provided: ", get_input_size());

    const auto& inDataShapeTensorShape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic input data shape");
    NODE_VALIDATION_CHECK(this, inDataShapeTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have input data shape tensor with rank 1, provided: ",
                          inDataShapeTensorShape.rank().get_length());

    const auto& outShapeDescriptorTensorShape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.is_static(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") doesn't support dynamic output shape descriptor");
    NODE_VALIDATION_CHECK(this, outShapeDescriptorTensorShape.rank().get_length() == 1,
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") must have output shape descriptor tensor with rank 1, provided: ",
                          outShapeDescriptorTensorShape.rank().get_length());

    const auto& inDataShapeTensorType = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          inDataShapeTensorType.is_static() &&
                          inDataShapeTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") input data type needs to be an integral type. Got: ",
                          inDataShapeTensorType);
    const auto& outShapeDescriptorTensorType = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          outShapeDescriptorTensorType.is_static() &&
                          outShapeDescriptorTensorType.is_integral_number(),
                          "OutShapeOfReshape (", get_friendly_name(),
                          ") shape descriptor type needs to be an integral type. Got: ",
                          outShapeDescriptorTensorType);

    set_output_type(0, element::i64, outShapeDescriptorTensorShape);
}

std::shared_ptr<Node> OutShapeOfReshape::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<OutShapeOfReshape>(new_args.at(0), new_args.at(1), m_specialZero);
}

bool OutShapeOfReshape::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("special_zero", m_specialZero);
    return true;
}

namespace {

template<element::Type_t ET>
bool getShapeFromHostTensorData(const HostTensorPtr& data, Shape& result) {
    using T = typename element_type_traits<ET>::value_type;
    T* dataPtr = data->get_data_ptr<ET>();
    if (!dataPtr) {
        return false;
    }
    size_t outputRank = data->get_shape()[0];

    for (int i = 0; i < outputRank; i++) {
        result.push_back(dataPtr[i]);
    }

    return true;
}

template<element::Type_t ET>
bool setShapeToHostTensorData(const HostTensorPtr& data, const Shape& shape) {
    using T = typename element_type_traits<ET>::value_type;
    T* dataPtr = data->get_data_ptr<ET>();
    if (!dataPtr) {
        return false;
    }
    size_t outputRank = data->get_shape()[0];

    for (int i = 0; i < outputRank; i++) {
        dataPtr[i] = shape[i];
    }
    return true;
}

bool evaluateOutShapeOfReshape(
        const HostTensorPtr& inDataShapeTensor,
        const HostTensorPtr& outShapeDescriptorTensor,
        bool specialZero,
        const HostTensorPtr& outShapeTensor) {
    if (!inDataShapeTensor || !outShapeDescriptorTensor || !outShapeTensor) {
        return false;
    }
    Shape inputShape;
    Shape outputShape;

    switch (inDataShapeTensor->get_element_type()) {
        case element::Type_t::i8:
            if (!getShapeFromHostTensorData<element::Type_t::i8>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::i16:
            if (!getShapeFromHostTensorData<element::Type_t::i16>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::i32:
            if (!getShapeFromHostTensorData<element::Type_t::i32>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::i64:
            if (!getShapeFromHostTensorData<element::Type_t::i64>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::u8:
            if (!getShapeFromHostTensorData<element::Type_t::u8>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::u16:
            if (!getShapeFromHostTensorData<element::Type_t::u16>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::u32:
            if (!getShapeFromHostTensorData<element::Type_t::u32>(inDataShapeTensor, inputShape)) return false;
            break;
        case element::Type_t::u64:
            if (!getShapeFromHostTensorData<element::Type_t::u64>(inDataShapeTensor, inputShape)) return false;
            break;
        default: return false;
    }

    switch (outShapeDescriptorTensor->get_element_type()) {
        case element::Type_t::i8:
            if (!getShapeFromHostTensorData<element::Type_t::i8>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::i16:
            if (!getShapeFromHostTensorData<element::Type_t::i16>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::i32:
            if (!getShapeFromHostTensorData<element::Type_t::i32>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::i64:
            if (!getShapeFromHostTensorData<element::Type_t::i64>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::u8:
            if (!getShapeFromHostTensorData<element::Type_t::u8>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::u16:
            if (!getShapeFromHostTensorData<element::Type_t::u16>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::u32:
            if (!getShapeFromHostTensorData<element::Type_t::u32>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        case element::Type_t::u64:
            if (!getShapeFromHostTensorData<element::Type_t::u64>(outShapeDescriptorTensor, outputShape)) return false;
            break;
        default: return false;
    }

    if (std::any_of(outputShape.begin(), outputShape.end(), [](int64_t value) { return value < -1; })) {
        return false;
    }

    int zeroDimsCount = std::count_if(outputShape.begin(), outputShape.end(),
                                      [](int64_t value) { return value == 0; });
    int negativeDimsCount = std::count_if(outputShape.begin(), outputShape.end(),
                                          [](int64_t value) { return value == -1; });
    if (negativeDimsCount > 1) {
        return false;
    }

    size_t outputRank = outputShape.size();

    if (!(zeroDimsCount && specialZero) && !negativeDimsCount) {
        if (shape_size(inputShape) != shape_size(outputShape)) {
            return false;
        }
    } else {
        int negativeDimIdx = -1;

        size_t inputTotalDimCount = shape_size(inputShape);
        size_t outputTotalDimCount = 1;


        // compute the output shape
        for (size_t i = 0; i < outputRank; i++) {
            if (outputShape[i] == 0 && specialZero) {
                // Copy input_shape[i] for zero values
                if (i > inputShape.size() - 1) {
                    return false;
                }
                outputShape[i] = inputShape[i];
                outputTotalDimCount *= inputShape[i];
            } else if (outputShape[i] == -1) {
                negativeDimIdx = i;
            } else {
                outputTotalDimCount *= outputShape[i];
            }
        }

        if (negativeDimIdx != -1) {
            // Infer size such that number of output elements matches
            // input elements
            if (outputTotalDimCount == 0) {
                if (inputTotalDimCount != 0) {
                    return false;
                }
                outputShape[negativeDimIdx] = 0;
            } else {
                if (inputTotalDimCount % outputTotalDimCount != 0) {
                    return false;
                }
                outputShape[negativeDimIdx] = inputTotalDimCount / outputTotalDimCount;
            }
        }
    }

    switch (outShapeTensor->get_element_type()) {
        case element::Type_t::i8:
            if (!setShapeToHostTensorData<element::Type_t::i8>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::i16:
            if (!setShapeToHostTensorData<element::Type_t::i16>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::i32:
            if (!setShapeToHostTensorData<element::Type_t::i32>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::i64:
            if (!setShapeToHostTensorData<element::Type_t::i64>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::u8:
            if (!setShapeToHostTensorData<element::Type_t::u8>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::u16:
            if (!setShapeToHostTensorData<element::Type_t::u16>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::u32:
            if (!setShapeToHostTensorData<element::Type_t::u32>(outShapeTensor, outputShape)) return false;
            break;
        case element::Type_t::u64:
            if (!setShapeToHostTensorData<element::Type_t::u64>(outShapeTensor, outputShape)) return false;
            break;
        default: return false;
    }

    return true;
}

}  // namespace

bool OutShapeOfReshape::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) {
    return evaluateOutShapeOfReshape(inputs[0], inputs[1], m_specialZero, outputs[0]);
}


}  // namespace op
}  // namespace vpu
}  // namespace ngraph
