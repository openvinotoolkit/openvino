//
// Copyright (C) 2018-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "argument_parse_helpers.hpp"

#include <iostream>
#include <sstream>
#include <string>

/**
 * @brief Parse a string of per-layer values into a map.
 * @param str Input string in format "layer1:value1;layer2:value2" or a single default value
 * @param defaultValue Default value to use if the string is a single number
 * @return Map of layer name to value
 * @example parsePerLayerValues("logits:0.03;pred_boxes:0.05", 1.0)
 *          returns {"logits": 0.03, "pred_boxes": 0.05}
 * @example parsePerLayerValues("0.01", 1.0) returns {"*": 0.01}
 */
utils::PerLayerValueMap utils::parsePerLayerValues(const std::string& str, double defaultValue)
{
    utils::PerLayerValueMap result;

    // Always store the default as the wildcard fallback so getValueForLayer
    // never needs a separate default parameter.
    result["*"] = defaultValue;

    if (str.empty()) {
        return result;
    }

    // Try to parse as a single number first
    try {
        double value = std::stod(str);
        result["*"] = value;
        return result;
    } catch (...) {
        // Not a single number, parse as key:value pairs
    }

    // Parse "layer1:value1;layer2:value2" format
    std::istringstream stream(str);
    std::string pair;

    while (std::getline(stream, pair, ';')) {
        size_t colonPos = pair.find(':');
        if (colonPos != std::string::npos) {
            std::string layerName = pair.substr(0, colonPos);
            std::string valueStr = pair.substr(colonPos + 1);

            // Trim whitespace
            layerName.erase(0, layerName.find_first_not_of(" \t"));
            layerName.erase(layerName.find_last_not_of(" \t") + 1);
            valueStr.erase(0, valueStr.find_first_not_of(" \t"));
            valueStr.erase(valueStr.find_last_not_of(" \t") + 1);

            try {
                double value = std::stod(valueStr);
                result[layerName] = value;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse value '" << valueStr << "' for layer '" << layerName << "'" << std::endl;
            }
        }
    }

    return result;
}

/**
 * @brief Get the threshold value for a specific layer.
 * @param valueMap Map of layer name to value (always contains "*" fallback when
 *        created via parsePerLayerValues)
 * @param layerName Name of the layer
 * @return The threshold value for the layer
 */
double utils::getValueForLayer(const PerLayerValueMap& valueMap, const std::string& layerName)
{
    // First try exact match
    auto it = valueMap.find(layerName);
    if (it != valueMap.end()) {
        return it->second;
    }

    // Fall back to wildcard (always present when map was created with parsePerLayerValues)
    it = valueMap.find("*");
    if (it != valueMap.end()) {
        return it->second;
    }

    // Should never be reached for properly initialised maps.
    return 0.0;
}
