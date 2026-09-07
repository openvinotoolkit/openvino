// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/gguf/builder/metadata.hpp"

#include <type_traits>

#include "builder/sdk/metadata_store.hpp"
#include "openvino/core/tensor_util.hpp"

namespace ov::frontend::gguf {
namespace {

const GGUFMetaData* find(const detail::MetadataStore& store, const std::string& key) {
    const auto it = store.map.find(key);
    return it == store.map.end() ? nullptr : &it->second;
}

template <typename T>
std::vector<T> tensor_values(const ov::Tensor& tensor) {
    const auto type = tensor.get_element_type();
    // Keep typed reads strict; OpenVINO handles numeric widths and element conversion.
    if (type == ov::element::string || type.is_dynamic() ||
        (!type.is_integral() && !(std::is_floating_point_v<T> && type.is_real())))
        return {};
    return ov::util::to_vector<T>(tensor).value_or(std::vector<T>{});
}

template <typename T>
std::optional<T> number(const GGUFMetaData* value) {
    if (!value)
        return std::nullopt;
    if (const auto* v = std::get_if<int>(value))
        return static_cast<T>(*v);
    if (const auto* v = std::get_if<float>(value))
        return static_cast<T>(*v);
    if (const auto* t = std::get_if<ov::Tensor>(value); t && t->get_size() == 1) {
        const auto values = tensor_values<T>(*t);
        if (!values.empty())
            return values.front();
    }
    return std::nullopt;
}

template <typename T>
std::optional<T> stored_value(const GGUFMetaData* value) {
    if (value)
        if (const auto* v = std::get_if<T>(value))
            return *v;
    return std::nullopt;
}

template <typename T>
std::vector<T> array(const GGUFMetaData* value) {
    if (value)
        if (const auto* t = std::get_if<ov::Tensor>(value))
            return tensor_values<T>(*t);
    return {};
}
}  // namespace

bool GgufMetadata::has(const std::string& key) const {
    return find(*m_store, key) != nullptr;
}

std::optional<int64_t> GgufMetadata::get_int(const std::string& key) const {
    return number<int64_t>(find(*m_store, key));
}

std::optional<double> GgufMetadata::get_float(const std::string& key) const {
    return number<double>(find(*m_store, key));
}

std::optional<bool> GgufMetadata::get_bool(const std::string& key) const {
    if (auto i = get_int(key))
        return *i != 0;
    return std::nullopt;
}

std::optional<std::string> GgufMetadata::get_str(const std::string& key) const {
    return stored_value<std::string>(find(*m_store, key));
}

std::vector<int64_t> GgufMetadata::get_int_array(const std::string& key) const {
    const auto* value = find(*m_store, key);
    if (const auto v = stored_value<std::vector<int32_t>>(value))
        return {v->begin(), v->end()};
    return array<int64_t>(value);
}

std::vector<double> GgufMetadata::get_float_array(const std::string& key) const {
    return array<double>(find(*m_store, key));
}

std::vector<std::string> GgufMetadata::get_str_array(const std::string& key) const {
    return stored_value<std::vector<std::string>>(find(*m_store, key)).value_or(std::vector<std::string>{});
}

std::string GgufMetadata::architecture() const {
    return get_str("general.architecture").value_or("");
}

}  // namespace ov::frontend::gguf
