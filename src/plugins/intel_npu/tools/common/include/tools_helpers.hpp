// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

struct InputInfo {
    ov::element::Type type;
    ov::PartialShape partialShape;
    ov::Shape dataShape;
    ov::Layout layout;
};
using InputsInfo = std::map<std::string, InputInfo>;

std::string parameterNameToTensorName(std::string& name, std::vector<ov::Output<ov::Node>>& inputs_info) {
    // Looking for a tensor name match
    for (const auto& port : inputs_info) {
        if (port.get_names().count(name) > 0) {
            return port.get_any_name();
        }
    }
    // Looking for a node name match
    for (const auto& port : inputs_info) {
        if (!port.get_names().empty() && name == port.get_node()->get_friendly_name()) {
            return port.get_any_name();
        }
    }
    throw std::runtime_error("Provided I/O name \"" + name +
                             "\" is not found neither in tensor names nor in nodes names.");
}

std::map<std::string, std::vector<std::string>> parseInputParameters(std::string& parameter_string,
                                                                     std::vector<ov::Output<ov::Node>>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::vector<std::string>> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    auto input_name = search_string.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[parameterNameToTensorName(input_name, input_info)].push_back(std::move(input_value));
        } else {
            for (auto& item : input_info) {
                return_value[item.get_any_name()].push_back(input_value);
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || (search_string.front() != ',' && search_string.front() != '['))
            break;
        if (search_string.front() == ',') {
            if (search_string.length() > 1)
                search_string = search_string.substr(1);
            else
                throw std::logic_error("Can't parse input parameter string, there is nothing after the comma " +
                                       parameter_string);
        }
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}

/**
 * @brief Checks the model for dynamism and ensures it is compatible with NPU.
 *
 * This function performs several checks on the model's parameters to ensure that
 * the model's shape and batch size are static and supported by NPU. If the shape
 * or batch size is dynamic, it throws an exception when the user provides the shape
 * or batch. Else, it forces the batch size to 1.
 *
 * @param model A shared pointer to the OpenVINO model to be checked.
 * @param shapeOrBatchGiven A boolean flag indicating whether the shape or batch size
 *                          is provided by the user. Default is true.
 *
 * @throws std::logic_error If the model's rank or shape is dynamic and not supported by NPU.
 */
void boundDynamicShape(std::shared_ptr<ov::Model>& model, bool shapeOrBatchGiven = true) {
    std::cout << "Checking model for dynamism..." << std::endl;
    for (auto&& item : model->get_parameters()) {
        auto shape = item->get_partial_shape();
        auto rank = shape.rank();
        if (shape.is_static() && !shapeOrBatchGiven) {
            continue;
        }
        if (rank.is_dynamic()) {
            throw std::logic_error("Rank \"" + rank.to_string() + "\" of the shape \"" + shape.to_string() +
                                   "\" is dynamic which is not supported by NPU");
        }
        auto layout = item->get_layout();
        // Add batch N = 1 if not found in layout, and user does not provide `override_model_batch_size` or `shape`
        if (!ov::layout::has_batch(layout) && !shapeOrBatchGiven) {
            std::cout << "WARNING: Batch layout not found. Inserting 'N' dimension = 1 to the layout." << std::endl;
            item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            layout = item->get_layout();
        }
        if (shape[ov::layout::batch_idx(layout)].is_dynamic()) {
            if (shapeOrBatchGiven) {
                throw std::logic_error("ERROR: Shape \"" + shape.to_string() + "\"" +
                                       " has dynamic batch size which is not supported by NPU\n");
            } else {
                std::cout << "WARNING: Shape \"" + shape.to_string() + "\"" +
                                 " has dynamic batch size which is not supported by NPU\n"
                                 "         Setting batch to 1 forcibly"
                          << std::endl;
                ov::set_batch(model, 1);
                // Get the shape again
                shape = item->get_partial_shape();
                if (shape.is_dynamic()) {
                    throw std::logic_error("Model's input shape \"" + shape.to_string() + "\"" +
                                           " is dynamic which is not supported by NPU");
                }
            }
        }
    }
}

/**
 * @brief Reshapes the model based on the provided shape string or batch size.
 *
 * This function reshapes the model's input parameters based on the user-provided shape string or
 * user-provided model batch size override. Shape string and override model batch size cannot be
 * specified together. If the shape string is provided, it is parsed and used to reshape the model.
 * If the override model batch size is provided, a shape string will be reconstructed using the user-
 * provided batch size.
 *
 * @param inputsInfo A vector of OpenVINO output nodes representing the model's inputs.
 * @param infoMap A map to store the reshaped input information.
 * @param model A shared pointer to the OpenVINO model to be reshaped.
 * @param shapeString A string representing the desired shape for the model's inputs.
 * @param overrideModelBatchSize An integer specifying the batch size to override the model's batch size.
 * @param device A string view representing the target device for the model.
 *
 * @throws std::logic_error If both shapeString and overrideModelBatchSize are specified, or if
 *                          the shape string contains multiple shapes for one input, or if the
 *                          model's shape is dynamic and not supported by the device.
 */
void reshape(ov::OutputVector inputsInfo,
             InputsInfo& infoMap,
             std::shared_ptr<ov::Model>& model,
             std::string& shapeString,
             int overrideModelBatchSize,
             std::string_view device) {
    std::vector<InputsInfo> infoMaps;

    // shape and override_model_batch_size cannot be specificed together
    if (!shapeString.empty() && overrideModelBatchSize != 1) {
        throw std::logic_error(R"(Incompatible params: "shape" and "override_model_batch_size")");
    }

    // If override_model_batch_size is specified (default is 1):
    // 1. Get the layout of the model's parameters and check for batch dimension
    // 2. If batch dimension is not found, insert 'N' dimension = 1 to the layout
    // 3. Set the shape at batch index to match overrideModelBatchSize
    // 4. Pass the shape string as `shapeString` to be used as if user specified shape
    if (overrideModelBatchSize != 1) {
        for (auto&& item : model->get_parameters()) {
            auto layout = item->get_layout();
            if (!ov::layout::has_batch(layout)) {
                std::cout << "WARNING: Batch layout not found. Inserting 'N' dimension = 1 to the layout before "
                          << "setting new model batch of " << overrideModelBatchSize << std::endl;
                item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            }
            // Get partial shape and layout again (in case it's changed after setting batch dimension)
            auto shape = item->get_partial_shape();
            layout = item->get_layout();
            // Set shape at batch index to match overrideModelBatchSize
            shape[ov::layout::batch_idx(layout)] = overrideModelBatchSize;
            shapeString = shape.to_string();
        }
    }

    // Processes either user-provided shape string, or shape string reconstructed
    // through override_model_batch_size above
    if (!shapeString.empty()) {
        std::map<std::string, std::vector<std::string>> shapesMap = parseInputParameters(shapeString, inputsInfo);

        for (auto& item : inputsInfo) {
            InputInfo info;
            auto name = item.get_any_name();

            if (!shapesMap.empty()) {
                if (shapesMap.count(name)) {
                    if (shapesMap.at(name).size() > 1) {
                        // Example: -shape input1[..][..]
                        throw std::logic_error("shape command line parameter doesn't support multiple "
                                               "shapes for one input.");
                    }
                    info.partialShape = shapesMap.at(name)[0];
                } else {
                    info.partialShape = item.get_partial_shape();
                }
            }
            infoMap[name] = std::move(info);
            infoMaps.push_back(infoMap);
        }
        std::map<std::string, ov::PartialShape> newShapes;
        for (auto& item : infoMaps) {
            for (auto& map : item) {
                if (!newShapes.count(map.first)) {
                    newShapes[map.first] = map.second.partialShape;
                }
            }
        }
        model->reshape(newShapes);
    } else {  // FLAGS_shape is empty
        if (device.find("NPU") != std::string::npos || device.find("IMD") != std::string::npos ||
            // FIXME: SIT on CPU also requires to bound dynamic shapes
            device.find("CPU") != std::string::npos || device.find("TEMPLATE") != std::string::npos) {
            boundDynamicShape(model, false);
        }
    }
}

void printInputAndOutputsInfoShort(const ov::Model& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& param : network.get_parameters()) {
        auto l = param->get_layout();
        std::cout << "    " << param->get_friendly_name() << " : " << param->get_element_type() << " / "
                  << param->get_layout().to_string() << " / " << param->get_partial_shape().to_string() << std::endl;
    }
    std::cout << "Network outputs:" << std::endl;
    for (auto&& result : network.get_results()) {
        std::cout << "    " << result->get_friendly_name() << " : " << result->get_element_type() << " / "
                  << result->get_layout().to_string() << " / " << result->get_output_partial_shape(0).to_string()
                  << std::endl;
    }
}
