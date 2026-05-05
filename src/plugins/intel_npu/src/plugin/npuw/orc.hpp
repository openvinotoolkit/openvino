// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {
namespace orc {

using TypeId = std::uint16_t;
using Version = std::uint16_t;
using SectionFlags = std::uint32_t;
using SchemaUUID = std::array<std::uint8_t, 16>;

enum class SectionFlag : SectionFlags {
    OPTIONAL = 1u << 0,
    LEAF = 1u << 1,  // payload contains raw bytes only, no child sections
};

constexpr SectionFlags operator|(SectionFlag lhs, SectionFlag rhs) {
    return static_cast<SectionFlags>(lhs) | static_cast<SectionFlags>(rhs);
}

constexpr SectionFlags operator|(SectionFlags lhs, SectionFlag rhs) {
    return lhs | static_cast<SectionFlags>(rhs);
}

constexpr bool has_flag(SectionFlags flags, SectionFlag flag) {
    return (flags & static_cast<SectionFlags>(flag)) != 0u;
}

class Stream;

namespace detail {

template <typename T, typename = void>
struct has_member_serialize : std::false_type {};

template <typename T>
struct has_member_serialize<T, std::void_t<decltype(std::declval<T&>().serialize(std::declval<Stream&>()))>>
    : std::true_type {};

}  // namespace detail

class Stream {
public:
    static Stream reader(std::istream& stream);
    static Stream memory_reader(const void* data, std::size_t size);
    static Stream writer(std::ostream& stream);

    bool input() const;
    bool output() const;
    bool memory() const;
    std::size_t remaining() const;

    // Dispatches to value.serialize(*this) when the type provides a member serialize,
    // otherwise falls back to the free-function serialize(stream, value) found via ADL.
    template <typename T>
    Stream& operator&(T&& value) {
        using plain_type = std::remove_const_t<std::remove_reference_t<T>>;
        auto& v = const_cast<plain_type&>(value);
        if constexpr (detail::has_member_serialize<plain_type>::value) {
            v.serialize(*this);
        } else {
            serialize(*this, v);
        }
        return *this;
    }

    void bytes(void* data, std::size_t size);

private:
    Stream() = default;

    std::istream* m_input = nullptr;
    std::ostream* m_output = nullptr;
    const std::byte* m_memory = nullptr;
    std::size_t m_memory_size = 0u;
    std::size_t m_memory_offset = 0u;
};

template <typename T, std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value, bool> = true>
void serialize(Stream& stream, T& value) {
    stream.bytes(&value, sizeof(value));
}

inline void serialize(Stream& stream, std::byte& value) {
    stream.bytes(&value, sizeof(value));
}

void serialize(Stream& stream, std::string& value);

template <typename T1, typename T2>
void serialize(Stream& stream, std::pair<T1, T2>& value) {
    stream & value.first & value.second;
}

template <typename T>
void serialize(Stream& stream, std::vector<T>& value) {
    if (stream.output()) {
        auto size = value.size();
        stream & size;
        for (std::size_t idx = 0; idx < value.size(); ++idx) {
            if constexpr (std::is_same_v<T, bool>) {
                bool element = value[idx];
                stream & element;
            } else {
                stream& value[idx];
            }
        }
        return;
    }

    value.clear();
    std::size_t size = 0u;
    stream & size;
    value.reserve(size);
    for (std::size_t idx = 0; idx < size; ++idx) {
        T element{};
        stream & element;
        value.push_back(std::move(element));
    }
}

template <typename T, std::size_t N>
void serialize(Stream& stream, std::array<T, N>& value) {
    for (auto& element : value) {
        stream & element;
    }
}

template <typename T>
void serialize(Stream& stream, std::optional<T>& value) {
    bool has_value = value.has_value();
    stream & has_value;
    if (!has_value) {
        value.reset();
        return;
    }

    if (stream.output()) {
        stream & value.value();
        return;
    }

    T unpacked{};
    stream & unpacked;
    value = std::move(unpacked);
}

template <typename T>
std::vector<std::byte> encode(const T& value) {
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    auto stream = Stream::writer(buffer);
    auto& mutable_value = const_cast<std::remove_const_t<T>&>(value);
    stream & mutable_value;

    const auto raw = buffer.str();
    std::vector<std::byte> out(raw.size());
    if (!raw.empty()) {
        std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
}

template <typename T>
T decode(const std::vector<std::byte>& bytes) {
    auto stream = Stream::memory_reader(bytes.data(), bytes.size());
    T out{};
    stream & out;
    if (stream.remaining() != 0u) {
        OPENVINO_THROW("ORC payload has trailing bytes");
    }
    return out;
}

struct SectionHeader {
    TypeId type = 0u;
    Version version = 0u;
    SectionFlags flags = 0u;
    std::uint64_t size = 0u;
};

struct Section {
    TypeId type = 0u;
    Version version = 0u;
    SectionFlags flags = 0u;
    std::vector<std::byte> payload;
    std::vector<Section> children;

    bool is_optional() const {
        return has_flag(flags, SectionFlag::OPTIONAL);
    }

    bool is_leaf() const {
        return has_flag(flags, SectionFlag::LEAF);
    }

    bool is_container() const {
        return !is_leaf();
    }

    static Section raw(TypeId type, Version version, std::vector<std::byte> payload, SectionFlags flags = 0u);
    static Section container(TypeId type, Version version, std::vector<Section> children, SectionFlags flags = 0u);
};

template <typename T>
Section make_payload_section(TypeId type, Version version, const T& value, SectionFlags flags = 0u) {
    return Section::raw(type, version, encode(value), flags);
}

// File-level metadata returned by is_orc().
struct OrcHeader {
    std::uint16_t version = 0u;
    SchemaUUID schema_uuid{};
};

void write_file(std::ostream& stream, const Section& root, const SchemaUUID& uuid);
Section read_file(std::istream& stream);

// Peeks at the stream to check whether it contains an ORC blob.
// Returns the file header (format_version + schema_uuid) on success,
// nullopt if the magic does not match.
// Does not consume any bytes — the stream position is restored on return.
std::optional<OrcHeader> is_orc(std::istream& stream);

class Schema {
public:
    using Loader = std::function<std::any(const Section&, const Schema&)>;

    struct LoadedChild {
        TypeId type = 0u;
        std::any value;
    };

    template <typename T, typename LoaderT>
    void register_loader(TypeId type, LoaderT&& loader) {
        if (m_entries.count(type) != 0u) {
            OPENVINO_THROW("ORC schema already has a loader for type ID ", type);
        }

        Entry entry;
        entry.type = std::type_index(typeid(T));
        entry.loader = [fn = std::forward<LoaderT>(loader)](const Section& section, const Schema& schema) -> std::any {
            return std::any(fn(section, schema));
        };
        m_entries.emplace(type, std::move(entry));
    }

    bool knows(TypeId type) const;
    std::any load_any(const Section& section) const;

    template <typename T>
    T load(const Section& section) const {
        auto value = load_any(section);
        auto* typed = std::any_cast<T>(&value);
        if (typed == nullptr) {
            OPENVINO_THROW("ORC loader returned unexpected type for section type ID ", section.type);
        }
        return std::move(*typed);
    }

    std::vector<LoadedChild> load_children(const Section& container) const;
    const Section* find_child(const Section& container, TypeId type) const;
    const Section& require_child(const Section& container, TypeId type) const;

private:
    struct Entry {
        std::type_index type = std::type_index(typeid(void));
        Loader loader;
    };

    std::unordered_map<TypeId, Entry> m_entries;
};

template <typename T, typename = void>
struct has_prev : std::false_type {};

template <typename T>
struct has_prev<T, std::void_t<typename T::Prev>> : std::true_type {};

template <typename Target, typename Source>
Target migrate_chain(Source source) {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<Target>>,
                                 std::remove_cv_t<std::remove_reference_t<Source>>>) {
        return Target(std::move(source));
    } else {
        static_assert(has_prev<Target>::value, "Target must define Prev to use migrate_chain");
        return Target(migrate_chain<typename Target::Prev>(std::move(source)));
    }
}

namespace detail {

template <typename T, typename = void>
struct has_version : std::false_type {};

template <typename T>
struct has_version<T, std::void_t<decltype(T::kVersion)>> : std::true_type {};

template <typename Target, typename Current>
Target load_versioned_payload_impl(const Section& section) {
    static_assert(has_version<Current>::value, "Versioned ORC payload types must define kVersion");
    if (section.version == Current::kVersion) {
        return migrate_chain<Target>(decode<Current>(section.payload));
    }
    OPENVINO_THROW("Unsupported ORC section version ", section.version, " for type ID ", section.type);
}

template <typename Target, typename Current, typename Next, typename... Rest>
Target load_versioned_payload_impl(const Section& section) {
    static_assert(has_version<Current>::value, "Versioned ORC payload types must define kVersion");
    if (section.version == Current::kVersion) {
        return migrate_chain<Target>(decode<Current>(section.payload));
    }
    return load_versioned_payload_impl<Target, Next, Rest...>(section);
}

}  // namespace detail

template <typename Target, typename... Versions>
Target load_versioned_payload(const Section& section) {
    static_assert(sizeof...(Versions) != 0u, "At least one versioned payload type must be provided");
    if (section.is_container()) {
        OPENVINO_THROW("Cannot decode a container section as a raw ORC payload");
    }
    return detail::load_versioned_payload_impl<Target, Versions...>(section);
}

}  // namespace orc
}  // namespace npuw
}  // namespace ov

// Injects the standard versioning boilerplate into a frozen versioned type.
// PrevType must already define kVersion; ThisType must be the enclosing struct name.
//
// Example:
//   struct MyTypeV2 : MyTypeV1 {
//       ORC_DECLARE_VERSION(MyTypeV2, MyTypeV1)
//       int new_field = 0;
//       void serialize(Stream& stream) { Prev::serialize(stream); stream & new_field; }
//   };
#define ORC_DECLARE_VERSION(ThisType, PrevType)                               \
    using Prev = PrevType;                                                    \
    static constexpr ::ov::npuw::orc::Version kVersion = 1u + Prev::kVersion; \
    ThisType() = default;                                                     \
    explicit ThisType(PrevType _prev_init) : PrevType(std::move(_prev_init)) {}
