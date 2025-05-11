// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils.h"
#include "reshape_utils.h"


/**
 * @brief Parse data shapes for model
 */
std::map<std::string, std::vector<size_t>> parseDataShapes(const std::string &shapeString) {
    std::map<std::string, std::vector<size_t>> data_shapes;
    // Parse input parameter string
    std::vector<std::string> inputsShapes = split(shapeString, '&');

    for (size_t i = 0; i < inputsShapes.size(); i++) {
        std::vector<std::string> curLayout = split(inputsShapes[i], '*');

        std::string curLayoutName = curLayout.at(0);
        std::vector<size_t> shape;

        try {
            for (auto &dim: split(curLayout.at(1), ','))
                shape.emplace_back(std::stoi(dim));
        } catch (const std::exception &ex) {
            std::cerr << "Parsing data shapes failed with exception:\n"
                      << ex.what() << "\n";
        }
        data_shapes[curLayoutName] = shape;
    }
    return data_shapes;
}


/**
 * @brief Parse input shapes for model reshape
 */
std::map<std::string, ov::PartialShape> parseReshapeShapes(const std::string &shapeString) {
    std::map<std::string, ov::PartialShape> reshape_info;
    // Parse input parameter string
    std::vector<std::string> inputsShapes = split(shapeString, '&');

    for (size_t i = 0; i < inputsShapes.size(); i++) {
        std::vector<std::string> curLayout = split(inputsShapes[i], '*');

        std::string curLayoutName = curLayout.at(0);
        std::vector<ov::Dimension> shape;

        for (auto& dim : split(curLayout.at(1), ',')) {
            try {
                if (dim == "?" || dim == "-1") {
                    shape.emplace_back(ov::Dimension::dynamic());
                }
                else {
                    const std::string range_divider = "..";
                    size_t range_index = dim.find(range_divider);
                    if (range_index != std::string::npos) {
                        std::string min = dim.substr(0, range_index);
                        std::string max = dim.substr(range_index + range_divider.length());
                        shape.emplace_back(ov::Dimension(std::stoi(min), std::stoi(max)));
                    } else {
                        shape.emplace_back(ov::Dimension(std::stoi(dim)));
                    }
                }
            } catch (const std::exception &ex) {
                std::cerr << "Parsing reshape shapes failed with exception:\n"
                          << ex.what() << "\n";
            }
        }
        reshape_info[curLayoutName] = ov::PartialShape(shape);
    }
    return reshape_info;
}


/**
 * @brief Split input string using specified delimiter.
          Return vector with input tensor information
 */
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}


/**
 * @brief  Getting tensor shapes. If tensor is dynamic, static shape from data info will be returned.
 */
ov::Shape getTensorStaticShape(ov::Output<const ov::Node> &input,
                               std::map<std::string, std::vector<size_t>> dataShape) {
    std::string name;
    try {
        name = input.get_any_name();
    } catch (const ov::Exception &iex) {
        // Attempt to get a name for a Tensor without names
    }

    ov::Shape inputShape;

    if (!input.get_partial_shape().is_dynamic()) {
        return input.get_shape();
    }
    else if (dataShape.count(name)) {
        for (size_t j = 0; j < dataShape[name].size(); j++)
            inputShape.emplace_back(dataShape[name][j]);
    }
    else {
        throw std::logic_error("Please provide static shape for " + name + "input using -data_shapes argument!");
    }
    return inputShape;
}


/**
 * @brief Return copy of inputs object before reshape
 */
std::vector<ov::Output<ov::Node>> getCopyOfDefaultInputs(std::vector<ov::Output<ov::Node>> defaultInputs) {
    std::vector<ov::Output<ov::Node>> inputsCopy;
    for (size_t i = 0; i < defaultInputs.size(); i++) {
        auto nodeCopy = defaultInputs[i].get_node()->clone_with_new_inputs({});
        auto inputNode = ov::Output<ov::Node>(nodeCopy);
        inputsCopy.push_back(inputNode);
    }
    return inputsCopy;
}


/**
 * @brief Fill infer_request tensors with random values. The model shape is set separately. (OV API 2)
 */
void fillTensorsWithSpecifiedShape(ov::InferRequest& infer_request, std::vector<ov::Output<const ov::Node>> &inputs,
                                   std::map<std::string, std::vector<size_t>> dataShape) {
    for (size_t i = 0; i < inputs.size(); i++) {
        ov::Tensor input_tensor;
        ov::Shape inputShape = getTensorStaticShape(inputs[i], dataShape);

        if (inputs[i].get_element_type() == ov::element::f32) {
            input_tensor = fillTensorRandomDynamic<float>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::f64) {
            input_tensor = fillTensorRandomDynamic<double>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::f16) {
            input_tensor = fillTensorRandomDynamic<short>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::i32) {
            input_tensor = fillTensorRandomDynamic<int32_t>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::i64) {
            input_tensor = fillTensorRandomDynamic<int64_t>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::u8) {
            input_tensor = fillTensorRandomDynamic<uint8_t>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::i8) {
            input_tensor = fillTensorRandomDynamic<int8_t>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::u16) {
            input_tensor = fillTensorRandomDynamic<uint16_t>(inputs[i], inputShape);
        } else if (inputs[i].get_element_type() == ov::element::i16) {
            input_tensor = fillTensorRandomDynamic<int16_t>(inputs[i], inputShape);
        } else {
            throw std::logic_error("Input precision is not supported for " + inputs[i].get_element_type().get_type_name());
        }
        infer_request.set_input_tensor(i, input_tensor);
    }
}
