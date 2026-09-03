// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace detail {
// Opaque holder for the parser's metadata table. Defined inside the frontend so this header does
// not drag in the internal GGUF metadata variant (or the quant/ headers that define it): an
// extension is meant to read metadata through the typed accessors below, not to pattern-match a
// std::variant whose alternatives are an implementation detail.
struct MetadataStore;
}  // namespace detail

// Read-only view of a GGUF file's KV metadata: the port of llama.cpp's `llama_model_loader` as a
// model file uses it, i.e. `ml.get_key(...)`.
//
// GGUF metadata is UNTRUSTED input, so nothing here throws on a type mismatch or a missing key:
// every accessor either reports absence (`std::optional` / empty vector / `false`) or substitutes
// a caller-supplied default. A file that omits a key an architecture needs, or stores it with an
// unexpected type, must not be able to abort conversion from inside a metadata read; the builder
// decides what is fatal.
//
// Numeric accessors are width-agnostic on purpose. GGUF writers are inconsistent about whether a
// given key is u32/i32/u64/f32, and a model file should not have to care, so get_int() accepts any
// integer width and get_float() accepts any float width (and any integer, widened).
class GGUF_FRONTEND_API GgufMetadata {
public:
    explicit GgufMetadata(const detail::MetadataStore& store) : m_store(&store) {}

    bool has(const std::string& key) const;

    // ---- typed reads; std::nullopt when the key is absent or holds an incompatible type ----
    std::optional<int64_t> get_int(const std::string& key) const;
    std::optional<double> get_float(const std::string& key) const;
    std::optional<bool> get_bool(const std::string& key) const;
    std::optional<std::string> get_str(const std::string& key) const;

    // Empty vector when the key is absent or is not an array of that kind.
    std::vector<int64_t> get_int_array(const std::string& key) const;
    std::vector<double> get_float_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;

    // ---- llama.cpp `ml.get_key(KEY, dst)` analogues ----
    // Assign into `dst` and return true when the key is present and readable; leave `dst` untouched
    // and return false otherwise. This is the shape a ported load_arch_hparams() body is written
    // in, so those lines survive the port with only the key spelling changed.
    bool get_key(const std::string& key, int32_t& dst) const;
    bool get_key(const std::string& key, uint32_t& dst) const;
    bool get_key(const std::string& key, int64_t& dst) const;
    bool get_key(const std::string& key, float& dst) const;
    bool get_key(const std::string& key, double& dst) const;
    bool get_key(const std::string& key, bool& dst) const;
    bool get_key(const std::string& key, std::string& dst) const;

    // Value of `key`, or `fallback` when it is absent/unreadable.
    int64_t get_key_or(const std::string& key, int64_t fallback) const;
    double get_key_or(const std::string& key, double fallback) const;
    bool get_key_or(const std::string& key, bool fallback) const;
    std::string get_key_or(const std::string& key, const std::string& fallback) const;

    // `general.architecture`, or "" when the file does not name one.
    std::string architecture() const;

private:
    const detail::MetadataStore* m_store;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
