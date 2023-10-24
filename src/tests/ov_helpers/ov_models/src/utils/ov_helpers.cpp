// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/utils/ov_helpers.hpp"

#include <cstring>
#include <memory>
#include <queue>
#include <vector>

#include "backend.hpp"
#include "common_test_utils/test_enums.hpp"
#include "ngraph/specialize_function.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ngraph {
namespace helpers {

ov::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    ov::OutputVector outs;
    std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<ov::Node>& n) {
        for (const auto& out_p : n->outputs()) {
            outs.push_back(out_p);
        }
    });
    return outs;
}

std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> interpreterFunction(
    const std::shared_ptr<ov::Model>& function,
    const std::vector<std::vector<std::uint8_t>>& inputs,
    const std::vector<ov::element::Type>& inputTypes) {
    auto backend = ov::runtime::Backend::create();

    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    OPENVINO_ASSERT(parametersNumber == inputsNumber,
                    "Got function (",
                    function->get_friendly_name(),
                    ") with ",
                    parametersNumber,
                    " parameters, but ",
                    inputsNumber,
                    " input blobs");
    if (!inputTypes.empty()) {
        OPENVINO_ASSERT(inputTypes.size() == inputsNumber,
                        "Got function (",
                        function->get_friendly_name(),
                        ") with ",
                        inputsNumber,
                        " inputs, but ",
                        inputTypes.size(),
                        " types");
    }

    ov::TensorVector inputTensors(parametersNumber);
    for (size_t i = 0; i < parametersNumber; ++i) {
        const auto& parameter = parameters[i];
        const auto& parameterIndex = function->get_parameter_index(parameter);
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        const auto& parameterSize = shape_size(parameterShape) * parameterType.size();

        auto input = inputs[parameterIndex];
        const auto inType = inputTypes.empty() ? element::undefined : inputTypes[i];

        if (inType != element::undefined && inType != parameterType) {
            input = ngraph::helpers::convertOutputPrecision(input, inType, parameterType, shape_size(parameterShape));
        }

        const auto& inputSize = input.size();
        OPENVINO_ASSERT(parameterSize == inputSize,
                        "Got parameter (",
                        parameter->get_friendly_name(),
                        ") of size ",
                        parameterSize,
                        " bytes, but corresponding input with index ",
                        parameterIndex,
                        " has ",
                        inputSize,
                        " bytes");

        auto tensor = backend->create_tensor(parameterType, parameterShape);
        std::memcpy(tensor.data(), input.data(), parameterSize);
        inputTensors[i] = tensor;
    }

    const auto& results = function->get_results();
    ov::TensorVector outputTensors(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputTensors[i] = ov::Tensor(results[i]->get_element_type(), {0});
    }

    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);
    std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> outputs(results.size());
    for (size_t resultIndex = 0; resultIndex < results.size(); resultIndex++) {
        auto& output = outputs[resultIndex];
        output.first = results[resultIndex]->get_element_type();
        const auto& outputTensor = outputTensors[resultIndex];
        output.second.resize((shape_size(outputTensor.get_shape()) * outputTensor.get_element_type().bitwidth() + 7) >>
                             3);
        std::memcpy(output.second.data(), outputTensors[resultIndex].data(), output.second.size());
    }

    return outputs;
}

std::vector<ov::Tensor> interpretFunction(const std::shared_ptr<Function>& function,
                                          const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs) {
    auto backend = ov::runtime::Backend::create();

    const auto& funcInputs = function->inputs();
    const auto& funcInputsNumber = funcInputs.size();
    const auto& inputsNumber = inputs.size();
    OPENVINO_ASSERT(funcInputsNumber == inputsNumber,
                    "Got function (",
                    function->get_friendly_name(),
                    ") with ",
                    funcInputsNumber,
                    " parameters, but ",
                    inputsNumber,
                    " input blobs");

    ov::TensorVector inputTensors(funcInputsNumber);
    for (size_t i = 0; i < funcInputsNumber; ++i) {
        const auto& input = funcInputs[i];
        const auto& inputShape = input.get_shape();
        const auto& inputType = input.get_element_type();
        const auto& inputSize = shape_size(inputShape) * inputType.size();

        auto inputIt =
            std::find_if(inputs.begin(), inputs.end(), [&input](std::pair<std::shared_ptr<ov::Node>, ov::Tensor> elem) {
                return elem.first->get_friendly_name() == input.get_node_shared_ptr()->get_friendly_name();
            });
        if (inputIt == inputs.end()) {
            throw std::runtime_error("Parameter: " + input.get_node_shared_ptr()->get_friendly_name() +
                                     " was not find in input parameters");
        }
        auto inputTensor = inputIt->second;

        const auto& inputTensorSize = inputTensor.get_byte_size();
        OPENVINO_ASSERT(inputSize == inputTensorSize,
                        "Got parameter (",
                        input.get_node_shared_ptr()->get_friendly_name(),
                        ") of size ",
                        inputSize,
                        " bytes, but corresponding input ",
                        " has ",
                        inputTensorSize,
                        " bytes");

        auto tensor = backend->create_tensor(inputType, inputShape);
        inputTensor.copy_to(tensor);
        inputTensors[i] = tensor;
    }

    const auto& results = function->get_results();
    ov::TensorVector outputTensors(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputTensors[i] = ov::Tensor(results[i]->get_element_type(), {0});
    }

    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);

    return outputTensors;
}

std::shared_ptr<Function> foldFunction(const std::shared_ptr<Function>& function,
                                       const std::vector<std::vector<std::uint8_t>>& inputs,
                                       const std::vector<ov::element::Type>& inputTypes) {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    OPENVINO_ASSERT(parametersNumber == inputsNumber,
                    "Got function (",
                    function->get_friendly_name(),
                    ") with ",
                    parametersNumber,
                    " parameters, but ",
                    inputsNumber,
                    " input blobs");
    if (!inputTypes.empty()) {
        OPENVINO_ASSERT(inputTypes.size() == inputsNumber,
                        "Got function (",
                        function->get_friendly_name(),
                        ") with ",
                        inputsNumber,
                        " inputs, but ",
                        inputTypes.size(),
                        " types");
    }

    std::vector<element::Type> paramElementTypes;
    std::vector<PartialShape> paramShapes;
    std::vector<std::vector<std::uint8_t>> vecTmpConvertedInputs;
    vecTmpConvertedInputs.reserve(inputs.size());

    std::vector<void*> inBuffers;
    inBuffers.reserve(inputs.size());

    for (size_t i = 0; i < parametersNumber; ++i) {
        const auto& param = parameters[i];
        paramElementTypes.emplace_back(param->get_element_type());
        paramShapes.emplace_back(param->get_shape());
        auto parameterIndex = function->get_parameter_index(param);
        auto& input = inputs[parameterIndex];

        const auto inpType = inputTypes.empty() ? element::undefined : inputTypes[i];

        if (inpType != element::undefined && inpType != paramElementTypes.back()) {
            vecTmpConvertedInputs.emplace_back(
                convertOutputPrecision(input, inpType, param->get_element_type(), shape_size(param->get_shape())));
            inBuffers.push_back(vecTmpConvertedInputs.back().data());
        } else {
            // const_cast added to satisfy specialize_function interface
            // which requires inputs as std::vector<void *>
            inBuffers.push_back(const_cast<std::uint8_t*>(input.data()));
        }
    }

    NGRAPH_SUPPRESS_DEPRECATED_START;
    const auto& foldedFunc = ngraph::specialize_function(function, paramElementTypes, paramShapes, inBuffers);
    NGRAPH_SUPPRESS_DEPRECATED_END;
    ov::pass::ConstantFolding().run_on_model(foldedFunc);
    for (const auto& op : foldedFunc->get_ops()) {
        OPENVINO_ASSERT(op::is_constant(op) || op::is_output(op) || op::is_parameter(op),
                        "Function was not fully folded to constant state!\n",
                        "At least one non constant node with type ",
                        op->get_type_name(),
                        " present in function.");
    }
    return foldedFunc;
}

std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> getConstData(
    const std::shared_ptr<Function>& function) {
    size_t numOutputs = function->get_output_size();
    std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> outputs(numOutputs);
    auto funcResults = function->get_results();
    for (size_t i = 0; i < numOutputs; i++) {
        outputs[i].first = funcResults[i]->get_element_type();
        const auto& output = function->output(i).get_node_shared_ptr();
        OPENVINO_ASSERT(output->inputs().size() == 1);
        auto parrentNode = output->input_value(0).get_node_shared_ptr();
        OPENVINO_ASSERT(op::is_constant(parrentNode),
                        "Function was not fully folded to constant state!\n",
                        "Parent node of one of results is not constant and has type ",
                        parrentNode->get_type_name());

        const auto data = std::dynamic_pointer_cast<opset1::Constant>(parrentNode)->get_data_ptr<std::uint8_t>();
        const auto dataSize = shape_size(parrentNode->get_shape()) * parrentNode->get_element_type().size();
        outputs[i].second.resize(dataSize);
        std::copy(data, data + dataSize, outputs[i].second.data());
    }
    return outputs;
}

namespace {

std::string toString(const NodeTypeInfo& typeInfo) {
    return std::string(typeInfo.name) + " ver. " + std::string(typeInfo.version_id);
}

void CompareShapes(const PartialShape& actual, const PartialShape& expected) {
    OPENVINO_ASSERT(actual.relaxes(expected) && actual.refines(expected),
                    "Functions compare: Different shape detected ",
                    actual,
                    " and ",
                    expected);
}

void CompareNodes(const Node& actual, const Node& expected) {
    const auto& actualType = actual.get_type_info();
    const auto& expectedType = expected.get_type_info();
    OPENVINO_ASSERT(actualType == expectedType,
                    "Functions compare: data types must be equal ",
                    toString(actualType),
                    " != ",
                    toString(expectedType));

    const auto& numActualInputs = actual.inputs().size();
    const auto& numExpectedInputs = expected.inputs().size();
    OPENVINO_ASSERT(numActualInputs == numExpectedInputs,
                    "Functions compare: numbers of inputs are different: ",
                    numActualInputs,
                    " and ",
                    numExpectedInputs);

    const auto& numActualOutputs = actual.outputs().size();
    const auto& numExpectedOutputs = expected.outputs().size();
    OPENVINO_ASSERT(numActualOutputs == numExpectedOutputs,
                    "Functions compare: numbers of outputs are different: ",
                    numActualOutputs,
                    " and ",
                    numExpectedOutputs);
}

}  // namespace

void CompareFunctions(const Function& actual, const Function& expected) {
    const auto& actualOrderedOps = actual.get_ordered_ops();
    const auto& expectedOrderedOps = expected.get_ordered_ops();

    OPENVINO_ASSERT(expectedOrderedOps.size() == actualOrderedOps.size(),
                    "Functions compare: expected and actual ops number should be equal "
                    "but got ",
                    expectedOrderedOps.size(),
                    " and ",
                    actualOrderedOps.size(),
                    " respectively");

    for (std::size_t i = 0; i < expectedOrderedOps.size(); i++) {
        const auto& expectedOp = expectedOrderedOps[i];
        const auto& actualOp = actualOrderedOps[i];

        CompareNodes(*actualOp, *expectedOp);
        for (std::size_t i = 0; i < actualOp->inputs().size(); ++i) {
            const auto& actualShape = actualOp->input(i).get_partial_shape();
            const auto& expectedShape = expectedOp->input(i).get_partial_shape();
            CompareShapes(actualShape, expectedShape);
        }

        for (std::size_t i = 0; i < actualOp->outputs().size(); ++i) {
            const auto& actualShape = actualOp->output(i).get_partial_shape();
            const auto& expectedShape = expectedOp->output(i).get_partial_shape();
            CompareShapes(actualShape, expectedShape);
        }
    }
}

std::shared_ptr<ov::Node> getNodeSharedPtr(const ov::NodeTypeInfo& type_info, const ov::OutputVector& outputVector) {
    for (const auto& it : get_available_opsets()) {
        const auto& opset = it.second();
        if (opset.contains_type(type_info)) {
            const auto node = std::shared_ptr<ov::Node>(opset.create(type_info.name));
            node->set_arguments(outputVector);
            node->validate_and_infer_types();
            return node;
        }
    }
    OPENVINO_THROW("supported opsets does not contain op with name: ",
                   type_info.name,
                   " version: ",
                   type_info.version_id);
}

bool is_tensor_iterator_exist(const std::shared_ptr<ov::Model>& func) {
    const auto& ops = func->get_ops();
    for (const auto& node : ops) {
        const auto& ti = std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(node);
        if (ti) {
            return true;
        }
    }
    return false;
}

namespace {
template <int Bitwidth,
          typename Value,
          typename In,
          typename std::enable_if<std::is_unsigned<Value>::value, bool>::type = true>
Value fix_sign(In v) {
    return v;
}
template <int Bitwidth,
          typename Value,
          typename In,
          typename std::enable_if<std::is_signed<Value>::value, bool>::type = true>
Value fix_sign(In v) {
    constexpr unsigned sign_bit = 1u << (Bitwidth - 1);
    const bool is_negative_number = v & sign_bit;
    return is_negative_number ? v | 0xFFF0 : v;
}

template <int Bitwidth, typename Value>
class LowPrecisionWrapper {
public:
    static constexpr int bitwidth = Bitwidth;
    static constexpr uint8_t value_mask = (1u << bitwidth) - 1u;
    static constexpr int elements_in_byte = 8 / bitwidth;

    LowPrecisionWrapper(uint8_t* data, int position) : data(data), position(position) {}

    operator Value() const {
        return fix_sign<Bitwidth, Value>(((*data) >> (position * bitwidth)) & value_mask);
    }

    LowPrecisionWrapper& operator=(Value v) {
        uint8_t masked_value = v & value_mask;
        *data &= ~(value_mask << (position * bitwidth));
        *data |= masked_value << (position * bitwidth);
        return *this;
    }

private:
    int position{elements_in_byte - 1};
    uint8_t* data;
};

template <int Bitwidth, typename Value>
class LowPrecisionWrapperToConst {
public:
    static constexpr int bitwidth = Bitwidth;
    static constexpr uint8_t value_mask = (1u << bitwidth) - 1u;
    static constexpr int elements_in_byte = 8 / bitwidth;

    LowPrecisionWrapperToConst(const uint8_t* data, int position) : data(data), position(position) {}

    operator Value() const {
        return fix_sign<Bitwidth, Value>(((*data) >> (position * bitwidth)) & value_mask);
    }

private:
    int position{elements_in_byte - 1};
    const uint8_t* data;
};

template <int Bitwidth, typename Value>
class LowPrecistionRange {
public:
    static constexpr int bitwidth = Bitwidth;
    static constexpr int elements_in_byte = 8 / bitwidth;

    LowPrecistionRange(uint8_t* data) : data(data) {}

    LowPrecisionWrapper<Bitwidth, Value> operator[](size_t index) const {
        const ptrdiff_t byte_offset = index / elements_in_byte;
        const int bit_position = elements_in_byte - 1 - (index % elements_in_byte);
        return {data + byte_offset, bit_position};
    }

    uint8_t* data;
};

template <int Bitwidth, typename Value>
class LowPrecistionConstRange {
public:
    static constexpr int bitwidth = Bitwidth;
    static constexpr int elements_in_byte = 8 / bitwidth;

    LowPrecistionConstRange(const uint8_t* data) : data(data) {}

    LowPrecisionWrapperToConst<Bitwidth, Value> operator[](size_t index) const {
        const ptrdiff_t byte_offset = index / elements_in_byte;
        const int bit_position = elements_in_byte - 1 - (index % elements_in_byte);
        return {data + byte_offset, bit_position};
    }

    const uint8_t* data;
};

template <element::Type_t FromType,
          typename std::enable_if<FromType != element::Type_t::u1 && FromType != element::Type_t::u4 &&
                                      FromType != element::Type_t::i4,
                                  bool>::type = true>
const fundamental_type_for<FromType>* cast_to(const uint8_t* data) {
    return reinterpret_cast<const fundamental_type_for<FromType>*>(data);
}

template <element::Type_t FromType,
          typename std::enable_if<FromType != element::Type_t::u1 && FromType != element::Type_t::u4 &&
                                      FromType != element::Type_t::i4,
                                  bool>::type = true>
fundamental_type_for<FromType>* cast_to(uint8_t* data) {
    return reinterpret_cast<fundamental_type_for<FromType>*>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::u1, bool>::type = true>
LowPrecistionConstRange<1, uint8_t> cast_to(const uint8_t* data) {
    return LowPrecistionConstRange<1, uint8_t>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::u1, bool>::type = true>
LowPrecistionRange<1, uint8_t> cast_to(uint8_t* data) {
    return LowPrecistionRange<1, uint8_t>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::u4, bool>::type = true>
LowPrecistionConstRange<4, uint8_t> cast_to(const uint8_t* data) {
    return LowPrecistionConstRange<4, uint8_t>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::u4, bool>::type = true>
LowPrecistionRange<4, uint8_t> cast_to(uint8_t* data) {
    return LowPrecistionRange<4, uint8_t>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::i4, bool>::type = true>
LowPrecistionConstRange<4, int8_t> cast_to(const uint8_t* data) {
    return LowPrecistionConstRange<4, int8_t>(data);
}

template <element::Type_t FromType, typename std::enable_if<FromType == element::Type_t::i4, bool>::type = true>
LowPrecistionRange<4, int8_t> cast_to(uint8_t* data) {
    return LowPrecistionRange<4, int8_t>(data);
}

template <element::Type_t FromType, element::Type_t ToType>
std::vector<std::uint8_t> convertPrecision(const std::vector<std::uint8_t>& buffer, const size_t elementsCount) {
    using fromPrec = fundamental_type_for<FromType>;
    using toPrec = fundamental_type_for<ToType>;

    const size_t min_buffer_size = [&] {
        element::Type from_type(FromType);
        if (from_type.bitwidth() >= 8) {
            return elementsCount * sizeof(fromPrec);
        }
        return from_type.bitwidth() * elementsCount / 8;
    }();

    OPENVINO_ASSERT(buffer.size() >= min_buffer_size, "avoid buffer overflow");

    constexpr auto elementSize = sizeof(toPrec);
    std::vector<std::uint8_t> convertedData(elementsCount * elementSize);

    auto src = cast_to<FromType>(buffer.data());
    auto dst = cast_to<ToType>(convertedData.data());
    for (size_t i = 0; i < elementsCount; i++) {
        dst[i] = static_cast<toPrec>(src[i]);
    }
    return convertedData;
}

template <element::Type_t FromType>
std::vector<std::uint8_t> convertPrecisionFrom(const std::vector<std::uint8_t>& output,
                                               const element::Type_t& toPrecision,
                                               const size_t elementsCount) {
    switch (toPrecision) {
    case element::Type_t::boolean: {
        return convertPrecision<FromType, element::Type_t::boolean>(output, elementsCount);
    }
    case element::Type_t::bf16: {
        return convertPrecision<FromType, element::Type_t::bf16>(output, elementsCount);
    }
    case element::Type_t::f16: {
        return convertPrecision<FromType, element::Type_t::f16>(output, elementsCount);
    }
    case element::Type_t::f32: {
        return convertPrecision<FromType, element::Type_t::f32>(output, elementsCount);
    }
    case element::Type_t::f64: {
        return convertPrecision<FromType, element::Type_t::f64>(output, elementsCount);
    }
    case element::Type_t::i4: {
        return convertPrecision<FromType, element::Type_t::i4>(output, elementsCount);
    }
    case element::Type_t::i8: {
        return convertPrecision<FromType, element::Type_t::i8>(output, elementsCount);
    }
    case element::Type_t::i16: {
        return convertPrecision<FromType, element::Type_t::i16>(output, elementsCount);
    }
    case element::Type_t::i32: {
        return convertPrecision<FromType, element::Type_t::i32>(output, elementsCount);
    }
    case element::Type_t::i64: {
        return convertPrecision<FromType, element::Type_t::i64>(output, elementsCount);
    }
    case element::Type_t::u1: {
        return convertPrecision<FromType, element::Type_t::u1>(output, elementsCount);
    }
    case element::Type_t::u4: {
        return convertPrecision<FromType, element::Type_t::u4>(output, elementsCount);
    }
    case element::Type_t::u8: {
        return convertPrecision<FromType, element::Type_t::u8>(output, elementsCount);
    }
    case element::Type_t::u16: {
        return convertPrecision<FromType, element::Type_t::u16>(output, elementsCount);
    }
    case element::Type_t::u32: {
        return convertPrecision<FromType, element::Type_t::u32>(output, elementsCount);
    }
    case element::Type_t::u64: {
        return convertPrecision<FromType, element::Type_t::u64>(output, elementsCount);
    }
    default:
        throw std::runtime_error(std::string("convertOutputPrecision can't convert from: ") +
                                 element::Type(FromType).get_type_name() +
                                 " to: " + element::Type(toPrecision).get_type_name());
    }
}

}  // namespace
std::vector<std::uint8_t> convertOutputPrecision(const std::vector<std::uint8_t>& output,
                                                 const element::Type_t& fromPrecision,
                                                 const element::Type_t& toPrecision,
                                                 const size_t elementsCount) {
    switch (fromPrecision) {
    case element::Type_t::boolean: {
        return convertPrecisionFrom<element::Type_t::boolean>(output, toPrecision, elementsCount);
    }
    case element::Type_t::bf16: {
        return convertPrecisionFrom<element::Type_t::bf16>(output, toPrecision, elementsCount);
    }
    case element::Type_t::f16: {
        return convertPrecisionFrom<element::Type_t::f16>(output, toPrecision, elementsCount);
    }
    case element::Type_t::f32: {
        return convertPrecisionFrom<element::Type_t::f32>(output, toPrecision, elementsCount);
    }
    case element::Type_t::f64: {
        return convertPrecisionFrom<element::Type_t::f64>(output, toPrecision, elementsCount);
    }
    case element::Type_t::i4: {
        return convertPrecisionFrom<element::Type_t::i4>(output, toPrecision, elementsCount);
    }
    case element::Type_t::i8: {
        return convertPrecisionFrom<element::Type_t::i8>(output, toPrecision, elementsCount);
    }
    case element::Type_t::i16: {
        return convertPrecisionFrom<element::Type_t::i16>(output, toPrecision, elementsCount);
    }
    case element::Type_t::i32: {
        return convertPrecisionFrom<element::Type_t::i32>(output, toPrecision, elementsCount);
    }
    case element::Type_t::i64: {
        return convertPrecisionFrom<element::Type_t::i64>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u1: {
        return convertPrecisionFrom<element::Type_t::u1>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u4: {
        return convertPrecisionFrom<element::Type_t::u4>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u8: {
        return convertPrecisionFrom<element::Type_t::u8>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u16: {
        return convertPrecisionFrom<element::Type_t::u16>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u32: {
        return convertPrecisionFrom<element::Type_t::u32>(output, toPrecision, elementsCount);
    }
    case element::Type_t::u64: {
        return convertPrecisionFrom<element::Type_t::u64>(output, toPrecision, elementsCount);
    }
    default:
        throw std::runtime_error(std::string("convertOutputPrecision can't convert from: ") +
                                 element::Type(fromPrecision).get_type_name() + " precision");
    }
}

std::ostream& operator<<(std::ostream& os, MemoryTransformation type) {
    switch (type) {
    case MemoryTransformation::NONE:
        os << "NONE";
        break;
    case MemoryTransformation::LOW_LATENCY_V2:
        os << "LOW_LATENCY_V2";
        break;
    case MemoryTransformation::LOW_LATENCY_V2_REGULAR_API:
        os << "LOW_LATENCY_V2_REGULAR_API";
        break;
    case MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT:
        os << "LOW_LATENCY_V2_ORIGINAL_INIT";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_TYPE");
    }
    return os;
}

void resize_function(std::shared_ptr<ov::Model> function, const std::vector<ov::Shape>& targetInputStaticShapes) {
    auto inputs = function->inputs();
    std::map<ov::Output<ov::Node>, ov::PartialShape> shapes;
    if (inputs.size() > targetInputStaticShapes.size()) {
        throw std::runtime_error("targetInputStaticShapes.size() = " + std::to_string(targetInputStaticShapes.size()) +
                                 " != inputs.size() = " + std::to_string(inputs.size()));
    }
    for (size_t i = 0; i < inputs.size(); i++) {
        shapes.insert({inputs[i], targetInputStaticShapes[i]});
    }
    function->reshape(shapes);
}

}  // namespace helpers
}  // namespace ngraph
