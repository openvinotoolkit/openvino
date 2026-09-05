// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the extension-facing metadata view.
//
// The parser stores every GGUF scalar as a shape-{} ov::Tensor (so numeric metadata round-trips
// through ov::element types) and arrays as a shape-{n} tensor or a vector<std::string>. Callers of
// this view should not have to know that, nor which of u8/u32/i32/u64 a particular writer chose for
// a particular key, so the readers below accept any compatible width and convert.

#include "openvino/frontend/gguf/builder/metadata.hpp"

#include <cstring>

#include "builder/sdk/metadata_store.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

// Read a shape-{} or single-element tensor as an integer, whatever width it was stored at.
// Returns false for a non-scalar or a float tensor.
bool tensor_as_int(const ov::Tensor& t, int64_t& out) {
    if (t.get_size() != 1) {
        return false;
    }
    const auto et = t.get_element_type();
    if (et == ov::element::i8) {
        out = *t.data<int8_t>();
    } else if (et == ov::element::u8) {
        out = *t.data<uint8_t>();
    } else if (et == ov::element::i16) {
        out = *t.data<int16_t>();
    } else if (et == ov::element::u16) {
        out = *t.data<uint16_t>();
    } else if (et == ov::element::i32) {
        out = *t.data<int32_t>();
    } else if (et == ov::element::u32) {
        out = *t.data<uint32_t>();
    } else if (et == ov::element::i64) {
        out = *t.data<int64_t>();
    } else if (et == ov::element::u64) {
        out = static_cast<int64_t>(*t.data<uint64_t>());
    } else if (et == ov::element::boolean) {
        out = *t.data<char>() != 0 ? 1 : 0;
    } else {
        return false;
    }
    return true;
}

bool tensor_as_float(const ov::Tensor& t, double& out) {
    if (t.get_size() != 1) {
        return false;
    }
    const auto et = t.get_element_type();
    if (et == ov::element::f32) {
        out = *t.data<float>();
        return true;
    }
    if (et == ov::element::f64) {
        out = *t.data<double>();
        return true;
    }
    int64_t i = 0;
    if (tensor_as_int(t, i)) {
        out = static_cast<double>(i);
        return true;
    }
    return false;
}

// Element `i` of an array tensor, as an integer.
bool element_as_int(const ov::Tensor& t, size_t i, int64_t& out) {
    const auto et = t.get_element_type();
    if (et == ov::element::i8) {
        out = t.data<int8_t>()[i];
    } else if (et == ov::element::u8) {
        out = t.data<uint8_t>()[i];
    } else if (et == ov::element::i16) {
        out = t.data<int16_t>()[i];
    } else if (et == ov::element::u16) {
        out = t.data<uint16_t>()[i];
    } else if (et == ov::element::i32) {
        out = t.data<int32_t>()[i];
    } else if (et == ov::element::u32) {
        out = t.data<uint32_t>()[i];
    } else if (et == ov::element::i64) {
        out = t.data<int64_t>()[i];
    } else if (et == ov::element::u64) {
        out = static_cast<int64_t>(t.data<uint64_t>()[i]);
    } else if (et == ov::element::boolean) {
        out = t.data<char>()[i] != 0 ? 1 : 0;
    } else {
        return false;
    }
    return true;
}

bool element_as_float(const ov::Tensor& t, size_t i, double& out) {
    const auto et = t.get_element_type();
    if (et == ov::element::f32) {
        out = t.data<float>()[i];
        return true;
    }
    if (et == ov::element::f64) {
        out = t.data<double>()[i];
        return true;
    }
    int64_t v = 0;
    if (element_as_int(t, i, v)) {
        out = static_cast<double>(v);
        return true;
    }
    return false;
}

}  // namespace

bool GgufMetadata::has(const std::string& key) const {
    return m_store->map.count(key) > 0;
}

std::optional<int64_t> GgufMetadata::get_int(const std::string& key) const {
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return std::nullopt;
    }
    if (const auto* v = std::get_if<int>(&it->second)) {
        return static_cast<int64_t>(*v);
    }
    if (const auto* v = std::get_if<float>(&it->second)) {
        return static_cast<int64_t>(*v);
    }
    if (const auto* t = std::get_if<ov::Tensor>(&it->second)) {
        int64_t out = 0;
        if (tensor_as_int(*t, out)) {
            return out;
        }
    }
    return std::nullopt;
}

std::optional<double> GgufMetadata::get_float(const std::string& key) const {
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return std::nullopt;
    }
    if (const auto* v = std::get_if<float>(&it->second)) {
        return static_cast<double>(*v);
    }
    if (const auto* v = std::get_if<int>(&it->second)) {
        return static_cast<double>(*v);
    }
    if (const auto* t = std::get_if<ov::Tensor>(&it->second)) {
        double out = 0;
        if (tensor_as_float(*t, out)) {
            return out;
        }
    }
    return std::nullopt;
}

std::optional<bool> GgufMetadata::get_bool(const std::string& key) const {
    if (auto i = get_int(key)) {
        return *i != 0;
    }
    return std::nullopt;
}

std::optional<std::string> GgufMetadata::get_str(const std::string& key) const {
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return std::nullopt;
    }
    if (const auto* v = std::get_if<std::string>(&it->second)) {
        return *v;
    }
    return std::nullopt;
}

std::vector<int64_t> GgufMetadata::get_int_array(const std::string& key) const {
    std::vector<int64_t> out;
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return out;
    }
    if (const auto* v = std::get_if<std::vector<int32_t>>(&it->second)) {
        out.assign(v->begin(), v->end());
        return out;
    }
    if (const auto* t = std::get_if<ov::Tensor>(&it->second)) {
        out.reserve(t->get_size());
        for (size_t i = 0; i < t->get_size(); ++i) {
            int64_t e = 0;
            if (!element_as_int(*t, i, e)) {
                return {};
            }
            out.push_back(e);
        }
    }
    return out;
}

std::vector<double> GgufMetadata::get_float_array(const std::string& key) const {
    std::vector<double> out;
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return out;
    }
    if (const auto* t = std::get_if<ov::Tensor>(&it->second)) {
        out.reserve(t->get_size());
        for (size_t i = 0; i < t->get_size(); ++i) {
            double e = 0;
            if (!element_as_float(*t, i, e)) {
                return {};
            }
            out.push_back(e);
        }
    }
    return out;
}

std::vector<std::string> GgufMetadata::get_str_array(const std::string& key) const {
    auto it = m_store->map.find(key);
    if (it == m_store->map.end()) {
        return {};
    }
    if (const auto* v = std::get_if<std::vector<std::string>>(&it->second)) {
        return *v;
    }
    return {};
}

bool GgufMetadata::get_key(const std::string& key, int32_t& dst) const {
    if (auto v = get_int(key)) {
        dst = static_cast<int32_t>(*v);
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, uint32_t& dst) const {
    if (auto v = get_int(key)) {
        dst = static_cast<uint32_t>(*v);
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, int64_t& dst) const {
    if (auto v = get_int(key)) {
        dst = *v;
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, float& dst) const {
    if (auto v = get_float(key)) {
        dst = static_cast<float>(*v);
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, double& dst) const {
    if (auto v = get_float(key)) {
        dst = *v;
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, bool& dst) const {
    if (auto v = get_bool(key)) {
        dst = *v;
        return true;
    }
    return false;
}

bool GgufMetadata::get_key(const std::string& key, std::string& dst) const {
    if (auto v = get_str(key)) {
        dst = *v;
        return true;
    }
    return false;
}

int64_t GgufMetadata::get_key_or(const std::string& key, int64_t fallback) const {
    return get_int(key).value_or(fallback);
}

double GgufMetadata::get_key_or(const std::string& key, double fallback) const {
    return get_float(key).value_or(fallback);
}

bool GgufMetadata::get_key_or(const std::string& key, bool fallback) const {
    return get_bool(key).value_or(fallback);
}

std::string GgufMetadata::get_key_or(const std::string& key, const std::string& fallback) const {
    return get_str(key).value_or(fallback);
}

std::string GgufMetadata::architecture() const {
    return get_str("general.architecture").value_or("");
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
