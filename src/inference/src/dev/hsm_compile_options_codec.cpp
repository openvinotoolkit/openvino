// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_compile_options_codec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/util/hash_util.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {
namespace {

constexpr char us = '\x1f';  // ends a key and its value
constexpr char gs = '\x1d';  // opens and closes a nested map
constexpr char rs = '\x1e';  // opens and closes a reserved block
constexpr char fs = '\x1c';  // ends the entries
constexpr std::string_view reserved_bytes{"\0\x1f\x1d\x1e\x1c", 5};

constexpr uint64_t hash_step(uint64_t hash, char byte) noexcept {
    return ov::util::u64_hash_combine(hash, static_cast<uint8_t>(byte));
}

// Sends every piece to the sink and folds it into the digest that closes the payload.
class Emitter {
public:
    explicit Emitter(const SectionSink& sink) noexcept : m_sink{sink} {}

    void put(std::string_view bytes) {
        if (!bytes.empty()) {
            for (const auto byte : bytes) {
                m_hash = hash_step(m_hash, byte);
            }
            m_sink(ov::util::as_memory_view(bytes));
        }
    }

    void put(char separator) {
        put(std::string_view{&separator, 1});
    }

    void finish() {
        put(fs);
        m_sink(ov::util::as_memory_view(m_hash));
    }

private:
    const SectionSink& m_sink;
    uint64_t m_hash = 0;
};

void check_text(std::string_view what, std::string_view text, std::string_view key) {
    OPENVINO_ASSERT(text.find_first_of(reserved_bytes) == std::string_view::npos,
                    "Compile option ",
                    what,
                    " contains a reserved byte (NUL, 0x1C-0x1F), key: ",
                    key);
}

template <class T>
bool set_precision(std::ostream& os, const ov::Any& value) {
    if (value.is<T>()) {
        os << std::setprecision(std::numeric_limits<T>::max_digits10);
        return true;
    }
    return false;
}

// Classic locale and max_digits10 keep the text deterministic and parseable back to the same value.
std::string to_option_string(const ov::Any& value) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    (void)(set_precision<float>(os, value) || set_precision<double>(os, value) ||
           set_precision<long double>(os, value));
    value.print(os);
    return os.str();
}

void write_key(Emitter& out, const std::string& key) {
    OPENVINO_ASSERT(!key.empty(), "Compile option key must not be empty");
    check_text("key", key, key);
    out.put(key);
}

void write_leaf(Emitter& out, const std::string& key, const std::string& text) {
    write_key(out, key);
    out.put(us);
    check_text("value", text, key);
    out.put(text);
    out.put(us);
}

void write_entries(Emitter& out, const ov::AnyMap& options);

void write_nested(Emitter& out, const std::string& key, const ov::AnyMap& options) {
    write_key(out, key);
    out.put(gs);
    write_entries(out, options);
    out.put(gs);
}

void write_nested(Emitter& out, const std::string& key, const std::map<std::string, ov::AnyMap>& options) {
    write_key(out, key);
    out.put(gs);
    for (const auto& [name, nested] : options) {
        write_nested(out, name, nested);
    }
    out.put(gs);
}

void write_entries(Emitter& out, const ov::AnyMap& options) {
    for (const auto& [key, value] : options) {
        OPENVINO_ASSERT(!value.empty(), "Compile option value must not be empty, key: ", key);

        if (value.is<std::string>()) {
            write_leaf(out, key, value.as<std::string>());
        } else if (value.is<ov::AnyMap>()) {
            write_nested(out, key, value.as<ov::AnyMap>());
        } else if (value.is<std::map<std::string, ov::AnyMap>>()) {
            write_nested(out, key, value.as<std::map<std::string, ov::AnyMap>>());
        } else {
            std::string text;
            try {
                text = to_option_string(value);
            } catch (const std::exception& e) {
                OPENVINO_THROW("Cannot convert compile option ", key, " to string: ", e.what());
            }
            OPENVINO_ASSERT(!text.empty(),
                            "Compile option ",
                            key,
                            " of type ",
                            value.type_info().name(),
                            " has no string form");
            write_leaf(out, key, text);
        }
    }
}

enum class Status { more, done, malformed };

// Byte-at-a-time parser; open nested maps live on a heap stack so depth is bounded by memory, not the call stack.
class Parser {
public:
    Parser() {
        m_levels.emplace_back();
    }

    Status consume(char byte) {
        switch (m_state) {
        case State::key:
            return on_key(byte);
        case State::value:
            return on_value(byte);
        default:
            return on_reserved(byte);
        }
    }

    ov::AnyMap take() {
        return std::move(m_levels.front().options);
    }

private:
    enum class State { key, value, reserved };

    struct Level {
        std::string name;
        ov::AnyMap options;
    };

    Status on_key(char byte) {
        switch (byte) {
        case us:
            m_state = State::value;
            return Status::more;
        case gs:
            return m_key.empty() ? close_level() : open_level();
        case rs:
            m_key.clear();
            m_state = State::reserved;
            return Status::more;
        case fs:
            return m_key.empty() && m_levels.size() == 1 ? Status::done : Status::malformed;
        case '\0':
            return Status::malformed;
        default:
            m_key.push_back(byte);
            return Status::more;
        }
    }

    Status on_value(char byte) {
        switch (byte) {
        case us:
            if (!m_key.empty() && !m_levels.back().options.emplace(m_key, ov::Any{std::move(m_value)}).second) {
                return Status::malformed;
            }
            m_key.clear();
            m_value.clear();
            m_state = State::key;
            return Status::more;
        case gs:
        case rs:
        case fs:
        case '\0':
            return Status::malformed;
        default:
            m_value.push_back(byte);
            return Status::more;
        }
    }

    Status on_reserved(char byte) {
        if (byte == fs) {
            return Status::malformed;
        } else if (byte == rs) {
            m_state = State::key;
        }
        return Status::more;
    }

    // The key is checked here, not when the map closes: nothing else can add to this level meanwhile.
    Status open_level() {
        if (m_levels.back().options.count(m_key) != 0) {
            return Status::malformed;
        }
        m_levels.push_back({std::move(m_key), {}});
        m_key.clear();
        return Status::more;
    }

    Status close_level() {
        if (m_levels.size() == 1) {
            return Status::malformed;
        }
        auto level = std::move(m_levels.back());
        m_levels.pop_back();
        m_levels.back().options.emplace(std::move(level.name), ov::Any{std::move(level.options)});
        return Status::more;
    }

    State m_state = State::key;
    std::string m_key;
    std::string m_value;
    std::vector<Level> m_levels;
};

constexpr size_t chunk_size = 4096;

std::optional<ov::AnyMap> decode(const Section& section) {
    const auto size = section.size();
    std::array<std::byte, chunk_size> chunk;
    Parser parser;
    uint64_t hash = 0;

    for (size_t offset = 0; offset < size;) {
        const auto count = std::min(chunk.size(), size - offset);
        if (!section.read(offset, count, chunk.data())) {
            return std::nullopt;
        }
        for (size_t i = 0; i < count; ++i) {
            const auto byte = static_cast<char>(chunk[i]);
            hash = hash_step(hash, byte);
            switch (parser.consume(byte)) {
            case Status::more:
                break;
            case Status::malformed:
                return std::nullopt;
            case Status::done: {
                const auto digest_offset = offset + i + 1;
                uint64_t digest = 0;
                const auto is_valid = size - digest_offset == sizeof(digest) &&
                                      section.read(digest_offset, sizeof(digest), &digest) && digest == hash;
                return is_valid ? std::optional{parser.take()} : std::nullopt;
            }
            }
        }
        offset += count;
    }
    return std::nullopt;
}

}  // namespace

void CompileOptionsEncoder::encode(const SectionSink& sink) const {
    Emitter out{sink};
    write_entries(out, m_options);
    out.finish();
}

std::optional<ov::AnyMap> decode_compile_options(const Section& section) noexcept {
    try {
        return decode(section);
    } catch (...) {  // CVS-198883 will address error handling
        return std::nullopt;
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
