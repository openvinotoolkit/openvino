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
struct MetadataAccess;
}  // namespace detail

// Read-only metadata view. Missing or incompatible scalar values return nullopt; callers
// choose defaults with std::optional::value_or or diagnose required keys. Numeric reads accept
// compatible widths, so extensions do not depend on the GGUF writer's integer/float precision.
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

    // `general.architecture`, or "" when the file does not name one.
    std::string architecture() const;

private:
    friend struct detail::MetadataAccess;
    const detail::MetadataStore* m_store;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
