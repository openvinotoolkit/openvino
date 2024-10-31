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
    auto count_name = std::any_of(inputs_info.begin(), inputs_info.end(), [name](ov::Output<ov::Node>& port) {
        return port.get_names().count(name) > 0;
    });
    if (count_name) {
        return name;
    } else {
        auto inputInfo = std::find_if(inputs_info.begin(), inputs_info.end(), [name](ov::Output<ov::Node>& port) {
            return name == port.get_node()->get_friendly_name();
        });
        if (inputInfo == inputs_info.end()) {
            throw std::runtime_error("Provided I/O name \"" + name +
                                     "\" is not found neither in tensor names nor in nodes names.");
        }
        return inputInfo->get_any_name();
    }
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
            return_value[parameterNameToTensorName(input_name, input_info)].push_back(input_value);
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

void boundDynamicShape(std::shared_ptr<ov::Model>& model) {
    for (auto&& item : model->get_parameters()) {
        auto shape = item->get_partial_shape();
        if (shape.is_static()) {
            continue;
        }
        auto rank = shape.rank();
        if (rank.is_dynamic()) {
            throw std::logic_error("Rank \"" + rank.to_string() + "\" of the shape \"" + shape.to_string() +
                                   "\" is dynamic which is not supported by NPU");
        }
        auto layout = item->get_layout();
        if (!ov::layout::has_batch(layout)) {
            item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            layout = item->get_layout();
        }
        if (shape[ov::layout::batch_idx(layout)].is_dynamic()) {
            std::cout << "WARNING: Shape \"" + shape.to_string() + "\"" +
                                 " has dynamic batch size which is not supported by NPU\n"
                                 "         Setting batch to 1 forcibly"
                      << std::endl;
            ov::set_batch(model, 1);
        }
        shape = item->get_partial_shape();
        if (shape.is_dynamic()) {
            throw std::logic_error("Model's input shape \"" + shape.to_string() + "\"" +
                                   " is dynamic which is not supported by NPU");
        }
    }
}

void setModelBatch(std::shared_ptr<ov::Model>& model, uint32_t batch = 1) {
    if (batch == 1) {
        return;
    }
    for (auto&& item : model->get_parameters()) {
        auto shape = item->get_partial_shape();
        auto rank = shape.rank();
        if (rank.is_dynamic()) {
            throw std::logic_error("Rank \"" + rank.to_string() + "\" of the shape \"" + shape.to_string() +
                                   "\" is dynamic which is not supported by NPU");
        }
        auto layout = item->get_layout();
        if (!ov::layout::has_batch(layout)) {
            item->set_layout(ov::Layout(layout.to_string().insert(1, "N,")));
            layout = item->get_layout();
        }
        if (shape[ov::layout::batch_idx(layout)].is_dynamic()) {
            throw std::logic_error("ERROR: Shape \"" + shape.to_string() + "\"" +
                                   " has dynamic batch size which is not supported by NPU\n"
                                   "Cannot apply fixed batch: " +
                                   std::to_string(batch) +
                                   ". Please remove the parameter from config: \"override_model_batch_size\"");
        }
        ov::set_batch(model, batch);
    }
}

void reshape(ov::OutputVector inputsInfo, InputsInfo& infoMap, std::shared_ptr<ov::Model>& model,
             std::string& shapeString, int overrideModelBatchSize, std::string_view device) {
    std::vector<InputsInfo> infoMaps;
    if (!shapeString.empty()) {
        std::map<std::string, std::vector<std::string>> shapesMap = parseInputParameters(shapeString, inputsInfo);

        if (overrideModelBatchSize != 1) {
            throw std::logic_error(R"(Incompatible params: "shape" and "override_model_batch_size")");
        }
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
    } else {
        if (device.find("NPU") != std::string::npos ||
            // FIXME: SIT on CPU also requires to bound dynamic shapes
            device.find("CPU") != std::string::npos || device.find("TEMPLATE") != std::string::npos) {
            boundDynamicShape(model);
        }

        setModelBatch(model, overrideModelBatchSize);
    }
}
