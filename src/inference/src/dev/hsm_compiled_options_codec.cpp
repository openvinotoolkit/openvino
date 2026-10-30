// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_compiled_options_codec.hpp"

#include <cstring>
#include <exception>
#include <ios>
#include <new>
#include <string>
#include <string_view>

#include "openvino/core/except.hpp"
#include "openvino/util/log.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
namespace {

using Length = uint32_t;

// resize + memcpy rather than vector::insert(), which trips GCC 11's -Wstringop-overread false positive.
void append(std::vector<std::byte>& out, const void* data, size_t size) {
    if (size != 0) {
        const auto offset = out.size();
        out.resize(offset + size);
        std::memcpy(out.data() + offset, data, size);
    }
}

template <class T>
void put(std::vector<std::byte>& out, const T& value) {
    append(out, &value, sizeof(T));
}

void put_bytes(std::vector<std::byte>& out, std::string_view bytes) {
    append(out, bytes.data(), bytes.size());
}

void put_header(std::vector<std::byte>& out, std::string_view key, CompiledOptionKind kind, Length value_len) {
    put(out, static_cast<Length>(key.size()));
    put_bytes(out, key);
    put(out, kind);
    put(out, value_len);
}

void check_size(const std::string& key, const std::vector<std::byte>& out) {
    OPENVINO_ASSERT(out.size() <= max_compiled_options_size,
                    "Compiled options exceed ",
                    max_compiled_options_size,
                    " bytes at key: ",
                    key);
}

void encode_map(std::vector<std::byte>& out, const ov::AnyMap& options, size_t depth) {
    OPENVINO_ASSERT(depth <= max_compiled_options_depth,
                    "Compiled options nest deeper than ",
                    max_compiled_options_depth,
                    " levels");
    put(out, static_cast<Length>(options.size()));

    for (const auto& [key, value] : options) {
        OPENVINO_ASSERT(!key.empty(), "Compiled option key must not be empty");
        OPENVINO_ASSERT(key.size() <= max_compiled_option_value_size,
                        "Compiled option key exceeds ",
                        max_compiled_option_value_size,
                        " bytes: ",
                        key.substr(0, 64));

        if (value.empty()) {
            put_header(out, key, CompiledOptionKind::empty, 0);
        } else if (value.is<ov::AnyMap>()) {
            put_header(out, key, CompiledOptionKind::map, 0);
            const auto value_len_pos = out.size() - sizeof(Length);
            encode_map(out, value.as<ov::AnyMap>(), depth + 1);
            const auto value_len = static_cast<Length>(out.size() - value_len_pos - sizeof(Length));
            std::memcpy(out.data() + value_len_pos, &value_len, sizeof(value_len));
        } else {
            std::string str;
            try {
                str = ov::Any{value}.as<std::string>();
            } catch (const std::exception& e) {
                OPENVINO_THROW("Cannot convert compiled option ", key, " to string: ", e.what());
            }
            OPENVINO_ASSERT(str.size() <= max_compiled_option_value_size,
                            "Compiled option value exceeds ",
                            max_compiled_option_value_size,
                            " bytes at key: ",
                            key);
            put_header(out, key, CompiledOptionKind::string, static_cast<Length>(str.size()));
            put_bytes(out, str);
        }
        check_size(key, out);
    }
}

class Cursor {
public:
    explicit Cursor(ov::util::MemoryView bytes) noexcept : m_pos(bytes.data()), m_end(bytes.data() + bytes.size()) {}

    template <class T>
    bool read(T& value) noexcept {
        if (remaining() < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, m_pos, sizeof(T));
        m_pos += sizeof(T);
        return true;
    }

    std::optional<ov::util::MemoryView> take(size_t size) noexcept {
        if (remaining() < size) {
            return std::nullopt;
        }
        const ov::util::MemoryView taken{m_pos, size};
        m_pos += size;
        return taken;
    }

    size_t remaining() const noexcept {
        return static_cast<size_t>(m_end - m_pos);
    }

private:
    const std::byte* m_pos;
    const std::byte* m_end;
};

std::string to_string(ov::util::MemoryView bytes) {
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::optional<ov::AnyMap> decode_map(ov::util::MemoryView bytes, size_t depth) {
    if (depth > max_compiled_options_depth) {
        return std::nullopt;
    }
    Cursor cursor{bytes};
    Length count{};
    constexpr size_t min_entry_size = 2 * sizeof(Length) + sizeof(CompiledOptionKind) + 1;
    if (!cursor.read(count) || count > cursor.remaining() / min_entry_size) {
        return std::nullopt;
    }

    ov::AnyMap options;
    for (Length i = 0; i < count; ++i) {
        Length key_len{};
        if (!cursor.read(key_len) || key_len == 0 || key_len > max_compiled_option_value_size) {
            return std::nullopt;
        }
        const auto key = cursor.take(key_len);
        CompiledOptionKind kind{};
        Length value_len{};
        if (!key || !cursor.read(kind) || !cursor.read(value_len)) {
            return std::nullopt;
        }
        const auto value = cursor.take(value_len);
        if (!value) {
            return std::nullopt;
        }

        ov::Any decoded;
        switch (kind) {
        case CompiledOptionKind::string:
            if (value_len > max_compiled_option_value_size) {
                return std::nullopt;
            }
            decoded = to_string(*value);
            break;
        case CompiledOptionKind::map:
            if (auto nested = decode_map(*value, depth + 1)) {
                decoded = std::move(*nested);
            } else {
                return std::nullopt;
            }
            break;
        case CompiledOptionKind::empty:
            if (value_len != 0) {
                return std::nullopt;
            }
            break;
        default:
            continue;
        }
        if (!options.emplace(to_string(*key), std::move(decoded)).second) {
            return std::nullopt;
        }
    }
    return cursor.remaining() == 0 ? std::optional{std::move(options)} : std::nullopt;
}

}  // namespace

std::vector<std::byte> encode_compiled_options(const ov::AnyMap& options) {
    std::vector<std::byte> bytes;
    encode_map(bytes, options, 1);
    return bytes;
}

std::optional<ov::AnyMap> decode_compiled_options(ov::util::MemoryView payload) {
    return !payload.empty() && payload.size() <= max_compiled_options_size ? decode_map(payload, 1) : std::nullopt;
}

std::optional<ov::AnyMap> decode_compiled_options(const Section& section) noexcept {
    try {
        if (section.size() > max_compiled_options_size) {
            return std::nullopt;
        } else if (const auto view = section.view(); !view.empty()) {
            return decode_map(view, 1);
        } else {
            std::vector<std::byte> bytes(section.size());
            return section.read(bytes.data()) ? decode_map({bytes.data(), bytes.size()}, 1) : std::nullopt;
        }
    } catch (const std::exception& exc) {
        OPENVINO_ERR("Failed to decode compile options: ", exc.what());
        return std::nullopt;
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
