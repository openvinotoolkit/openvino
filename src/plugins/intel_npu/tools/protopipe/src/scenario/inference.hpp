//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

template <typename T>
using AttrMap = std::map<std::string, T>;
// NB: This type is supposed to be used to hold in/out layers
// attributes such as precision, layout, shape etc.
//
// User can provide attributes either:
// 1. std::monostate - No value specified explicitly.
// 2. Attr - value specified explicitly that should be broadcasted to all layers.
// 3. AttrMap[str->T] - map specifies value for particular layer.
template <typename Attr>
using LayerVariantAttr = std::variant<std::monostate, AttrMap<Attr>, Attr>;

// NB: Map of model tag -> LayerVariantAttr<T>
template <typename T>
using ModelsAttrMap = std::unordered_map<std::string, LayerVariantAttr<T>>;

struct LayerInfo {
    std::string name;
    std::vector<int> dims;
    int prec;
};
using LayersInfo = std::vector<LayerInfo>;

std::vector<std::string> extractLayerNames(const std::vector<LayerInfo>& layers);

template <typename K, typename V>
std::optional<V> lookUp(const std::map<K, V>& map, const K& key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return {};
    }
    return std::make_optional(std::move(it->second));
}

template <typename T>
static AttrMap<T> unpackLayerAttr(const LayerVariantAttr<T>& attr, const std::vector<std::string>& layer_names,
                                  const std::string& attrname) {
    AttrMap<T> attrmap;
    if (std::holds_alternative<T>(attr)) {
        auto value = std::get<T>(attr);
        for (const auto& name : layer_names) {
            attrmap.emplace(name, value);
        }
    } else if (std::holds_alternative<AttrMap<T>>(attr)) {
        attrmap = std::get<AttrMap<T>>(attr);
        std::unordered_set<std::string> layers_set{layer_names.begin(), layer_names.end()};
        for (const auto& [name, attr] : attrmap) {
            const auto it = layers_set.find(name);
            if (it == layers_set.end()) {
                throw std::logic_error("Failed to find layer \"" + name + "\" to specify " + attrname);
            }
        }
    }
    return attrmap;
}

struct OpenVINOParams {
    struct ModelPath {
        std::string model;
        std::string bin;
    };
    struct BlobPath {
        std::string blob;
    };
    using Path = std::variant<ModelPath, BlobPath>;

    // NB: Mandatory parameters
    Path path;
    std::string device;
    // NB: Optional parameters
    LayerVariantAttr<int> input_precision;
    LayerVariantAttr<int> output_precision;
    LayerVariantAttr<std::string> input_layout;
    LayerVariantAttr<std::string> output_layout;
    LayerVariantAttr<std::string> input_model_layout;
    LayerVariantAttr<std::string> output_model_layout;
    std::map<std::string, std::string> config;
    size_t nireq = 1u;
};

struct ONNXRTParams {
    std::string model_path;
    std::map<std::string, std::string> session_options;
    // TODO: Extend for other available ONNXRT EP (e.g DML, CoreML, TensorRT, etc)
    struct OpenVINO {
        std::map<std::string, std::string> params_map;
    };
    // NB: std::monostate stands for the default MLAS Execution provider
    using EP = std::variant<std::monostate, OpenVINO>;
    std::optional<int> opt_level;
    EP ep;
};

using InferenceParams = std::variant<std::monostate, OpenVINOParams, ONNXRTParams>;
using InferenceParamsMap = std::unordered_map<std::string, InferenceParams>;
