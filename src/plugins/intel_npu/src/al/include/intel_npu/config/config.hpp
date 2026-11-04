// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cctype>
#include <chrono>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

template <class T>
struct TypePrinter {
    static constexpr bool hasName() {
        return false;
    }
    static constexpr const char* name();
};

#define TYPE_PRINTER(type)                    \
    template <>                               \
    struct TypePrinter<type> {                \
        static constexpr bool hasName() {     \
            return true;                      \
        }                                     \
        static constexpr const char* name() { \
            return #type;                     \
        }                                     \
    };

TYPE_PRINTER(bool)
TYPE_PRINTER(char)
TYPE_PRINTER(char*)
TYPE_PRINTER(int)
TYPE_PRINTER(unsigned int)
TYPE_PRINTER(int64_t)
TYPE_PRINTER(double)
TYPE_PRINTER(std::string)
TYPE_PRINTER(std::size_t)

#ifndef ONEAPI_MAKE_VERSION
/// @brief Generates generic 'oneAPI' API versions
#    define ONEAPI_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))

/// @brief extract 'oneAPI' API major version
#    define ONEAPI_VERSION_MAJOR(_version) ((_version) >> 16)

/// @brief extract 'oneAPI' API minor version
#    define ONEAPI_VERSION_MINOR(_version) ((_version) & 0x0000ffff)

#endif  // ONEAPI_MAKE_VERSION

//
// OptionParser
//

namespace details {

template <typename T, typename = void>
struct IsIStreamable : std::false_type {};

template <typename T>
struct IsIStreamable<T, std::void_t<decltype(std::declval<std::istream&>() >> std::declval<T&>())>> : std::true_type {};

// NB: detecting `operator<<` instead would be useless for unscoped enums, since those promote to their
// underlying integral type and therefore always match one of the built-in `operator<<` overloads.
template <typename T, typename = void>
struct HasStringifyEnum : std::false_type {};

template <typename T>
struct HasStringifyEnum<T, std::void_t<decltype(stringifyEnum(std::declval<const T&>()))>> : std::true_type {};

// `operator>>` stops at the first character it can't consume, so a successfully parsed prefix alone is not
// enough to accept a value. Returns false when anything besides trailing whitespace is left in the stream.
inline bool isFullyConsumed(std::istream& stream) {
    for (auto next = stream.peek(); next != std::istream::traits_type::eof(); next = stream.peek()) {
        if (!std::isspace(static_cast<unsigned char>(next))) {
            return false;
        }
        stream.ignore();
    }

    return true;
}

}  // namespace details

// Default parser, relies on the `operator>>` declared for the value type. All the `ov::` property enums and
// wrappers provide one, so enum options usually don't need any parsing code of their own. Specialize this
// template or override `parse()` inside the option for types which have no such operator.
template <typename T>
struct OptionParser {
    static T parse(std::string_view val) {
        static_assert(details::IsIStreamable<T>::value,
                      "No `operator>>` is available for the option type, please provide either an `OptionParser` "
                      "specialization or a `parse()` implementation inside the option");

        std::istringstream stream{std::string(val)};
        T res{};
        stream >> res;
        OPENVINO_ASSERT(!stream.fail(), "Value '", val, "' is not a valid option value");
        // Reject values whose prefix alone is valid, e.g. "PLUGIN garbage" for an enum option
        OPENVINO_ASSERT(details::isFullyConsumed(stream), "Value '", val, "' is not a valid option value");

        return res;
    }
};

template <>
struct OptionParser<std::string> final {
    static std::string parse(std::string_view val) {
        return {val.data(), val.size()};
    }
};

template <>
struct OptionParser<bool> final {
    static bool parse(std::string_view val);
};

template <>
struct OptionParser<int32_t> final {
    static int32_t parse(std::string_view val);
};

template <>
struct OptionParser<uint32_t> final {
    static uint32_t parse(std::string_view val);
};

template <>
struct OptionParser<int64_t> final {
    static int64_t parse(std::string_view val);
};

template <>
struct OptionParser<uint64_t> final {
    static uint64_t parse(std::string_view val);
};

template <>
struct OptionParser<double> final {
    static double parse(std::string_view val);
};

void splitAndApply(const std::string& str, char delim, std::function<void(std::string_view)> callback);

template <typename T>
struct OptionParser<std::vector<T>> final {
    static std::vector<T> parse(std::string_view val) {
        std::vector<T> res;
        std::string val_str(val);
        splitAndApply(val_str, ',', [&](std::string_view item) {
            res.push_back(OptionParser<T>::parse(item));
        });
        return res;
    }
};

template <typename K, typename V>
struct OptionParser<std::map<K, V>> final {
    static std::map<K, V> parse(std::string_view val) {
        std::map<K, V> res;
        std::string val_str(val);
        splitAndApply(val_str, ',', [&](std::string_view item) {
            auto kv_delim_pos = item.find(":");
            OPENVINO_ASSERT(kv_delim_pos != std::string::npos);
            K key = OptionParser<K>::parse(std::string_view(item.substr(0, kv_delim_pos)));
            V value = OptionParser<V>::parse(std::string_view(item.substr(kv_delim_pos + 1)));
            res[key] = std::move(value);
        });
        return res;
    }
};

template <typename Rep, typename Period>
struct OptionParser<std::chrono::duration<Rep, Period>> final {
    static std::chrono::duration<Rep, Period> parse(std::string_view val) {
        std::istringstream stream{std::string(val)};

        Rep count{};
        // Anything left in the stream means the value is malformed, e.g. "12oops"
        if (stream >> count && details::isFullyConsumed(stream)) {
            OPENVINO_ASSERT(count >= 0,
                            "Value '",
                            count,
                            "' is not a valid time duration, non-negative values expected");
            return std::chrono::duration<Rep, Period>(count);
        }

        OPENVINO_THROW("Can't parse '", val, "' as time duration");
    }
};

//
// OptionPrinter
//

template <typename T>
struct OptionPrinter final {
    static std::string toString(const T& val) {
        std::stringstream ss;
        if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
            ss << std::fixed << std::setprecision(2) << val;
        } else if constexpr (std::is_enum_v<std::decay_t<T>> && details::HasStringifyEnum<std::decay_t<T>>::value) {
            // A `stringifyEnum` overload wins over an `operator<<`, enums are expected to provide either of the two
            ss << stringifyEnum(val);
        } else {
            ss << val;
        }
        return ss.str();
    }
};

template <typename K, typename V>
struct OptionPrinter<std::map<K, V>> final {
    static std::string toString(const std::map<K, V>& val) {
        std::stringstream ss;
        std::size_t counter = 0;
        std::size_t size = val.size();
        for (auto& [key, value] : val) {
            std::string key_str = OptionPrinter<K>::toString(key);
            std::string value_str = OptionPrinter<V>::toString(value);
            ss << key_str << ":" << value_str;
            if (counter < size - 1) {
                ss << ",";
            }
            ++counter;
        }
        return ss.str();
    }
};

// NB: boolean config option has values YES for true, NO for false
template <>
struct OptionPrinter<bool> final {
    static std::string toString(bool val);
};

template <typename Rep, typename Period>
struct OptionPrinter<std::chrono::duration<Rep, Period>> final {
    static std::string toString(const std::chrono::duration<Rep, Period>& val) {
        return std::to_string(val.count());
    }
};

//
// OptionMode
//

enum class OptionMode {
    Both,
    CompileTime,
    RunTime,
};

std::string_view stringifyEnum(OptionMode val);

//
// OptionBase
//

// Actual Option description must inherit this class and pass itself as template parameter.
template <class ActualOpt, typename T>
struct OptionBase {
    using ValueType = T;

    // `ActualOpt` must implement the following method:
    // static std::string_view key()

    static constexpr std::string_view getTypeName() {
        if constexpr (TypePrinter<T>::hasName()) {
            return TypePrinter<T>::name();
        }
        static_assert(TypePrinter<T>::hasName(),
                      "Options type is not a standard type, please add `getTypeName()` to your option");
    }

    // Overload this to provide environment variable support.
    static std::string_view envVar() {
        return "";
    }

    // Overload this to provide deprecated keys names.
    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    // Overload this to provide default value if it wasn't specified by user.
    // If it is std::nullopt - exception will be thrown in case of missing option access.
    static std::optional<T> defaultValue() {
        return std::nullopt;
    }

    // Overload this to provide more specific parser.
    static ValueType parse(std::string_view val) {
        return OptionParser<ValueType>::parse(val);
    }

    // Overload this to provide more specific validation
    static void validateValue(const ValueType&) {}

    // Overload this to provide more specific implementation.
    static OptionMode mode() {
        return OptionMode::Both;
    }

    static std::string toString(const ValueType& val) {
        return OptionPrinter<ValueType>::toString(val);
    }
};

//
// OptionValue
//

namespace details {

class OptionValue {
public:
    virtual ~OptionValue();

    virtual std::string_view getTypeName() const = 0;
    virtual std::string toString() const = 0;
};

template <typename Opt, typename T>
class OptionValueImpl final : public OptionValue {
    using ToStringFunc = std::string (*)(const T&);

public:
    template <typename U>
    OptionValueImpl(U&& val, ToStringFunc toStringImpl) : _val(std::forward<U>(val)),
                                                          _toStringImpl(toStringImpl) {}

    std::string_view getTypeName() const override final {
        if constexpr (TypePrinter<T>::hasName()) {
            return TypePrinter<T>::name();
        } else {
            return Opt::getTypeName();
        }
    }

    const T& getValue() const {
        return _val;
    }

    std::string toString() const override {
        return _toStringImpl(_val);
    }

private:
    T _val;
    ToStringFunc _toStringImpl = nullptr;
};

}  // namespace details

//
// OptionConcept
//

namespace details {

struct OptionConcept final {
    std::string_view (*key)() = nullptr;
    std::string_view (*envVar)() = nullptr;
    OptionMode (*mode)() = nullptr;
    std::shared_ptr<OptionValue> (*validateAndParse)(const ov::Any& val) = nullptr;
};

// `ov::Any::as<std::string>()` is lenient where the option parsers are not: it yields an empty string both
// for an empty `ov::Any` and for a payload whose type provides neither `operator<<` nor an
// `ov::util::Write` specialization (its `print()` is then a no-op). Parsers which accept an empty spelling as
// "nothing to parse" (`std::string`, `std::vector`, `std::map`) would silently turn such a payload into an
// empty option value, so reject it here instead. An empty spelling is only let through when it really was
// given as a string.
inline std::string stringifyForParsing(const ov::Any& val) {
    OPENVINO_ASSERT(!val.empty(), "No value was provided");

    auto valAsString = val.as<std::string>();
    OPENVINO_ASSERT(!valAsString.empty() || val.is<std::string>(),
                    "Value of type '",
                    val.type_info().name(),
                    "' can't be converted to a string");

    return valAsString;
}

template <class Opt>
std::shared_ptr<OptionValue> validateAndParse(const ov::Any& val) {
    using ValueType = typename Opt::ValueType;

    try {
        // Only a payload which already has the option's exact type is taken as is. Everything else - the string
        // spellings coming from `update()` as well as any other type - is routed through the option's own
        // parser, so that string based and `ov::Any` based updates accept exactly the same values and any
        // custom `parse()` is honored. `ov::Any::as<ValueType>()` would otherwise perform an unchecked
        // arithmetic conversion (e.g. -2.0f silently wrapping around into a `uint32_t`, which then passes the
        // option validation) or re-parse a string on its own, through `operator>>`, bypassing the option
        // entirely. For options which are themselves strings the parser is an identity conversion.
        auto parsedVal = val.is<ValueType>() ? val.as<ValueType>() : Opt::parse(stringifyForParsing(val));
        Opt::validateValue(parsedVal);
        return std::make_shared<OptionValueImpl<Opt, ValueType>>(std::move(parsedVal), &Opt::toString);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to parse '", Opt::key().data(), "' option : ", e.what());
    }
}

template <class Opt>
OptionConcept makeOptionModel() {
    return {&Opt::key, &Opt::envVar, &Opt::mode, &validateAndParse<Opt>};
}

}  // namespace details

//
// OptionsDesc
//

class OptionsDesc final {
public:
    template <class Opt>
    void add();

    bool has(std::string_view key) const;

    void reset();

    details::OptionConcept get(std::string_view key) const;
    void walk(std::function<void(const details::OptionConcept&)> cb) const;

private:
    std::unordered_map<std::string, details::OptionConcept> _impl;
    std::unordered_map<std::string, std::string> _deprecated;
    Logger _log{Logger::global().clone("OptionsDesc")};
};

template <class Opt>
void OptionsDesc::add() {
    OPENVINO_ASSERT(_impl.count(Opt::key().data()) == 0, "Option '", Opt::key().data(), "' was already registered");
    _impl.insert({Opt::key().data(), details::makeOptionModel<Opt>()});

    for (const auto& deprecatedKey : Opt::deprecatedKeys()) {
        OPENVINO_ASSERT(_deprecated.count(deprecatedKey.data()) == 0,
                        "Option '",
                        deprecatedKey.data(),
                        "' was already registered");
        _deprecated.insert({deprecatedKey.data(), Opt::key().data()});
    }
}

//
// Config
//

class Config final {
public:
    using ConfigMap = std::map<std::string, std::string>;

    /**
     * @brief Constructs a configuration bound to the given options descriptor.
     * @param desc Descriptor holding the set of options accepted by this configuration. Must not be null.
     */
    explicit Config(const std::shared_ptr<const OptionsDesc>& desc);

    /**
     * @brief Parses and stores all the given key/value pairs, overwriting previously set values.
     * @param options Map of option keys to their string representation.
     */
    void update(const ConfigMap& options);

    /**
     * @brief Parses and stores a single option value, overwriting a previously set one.
     * @param key The key of the option to set.
     * @param value The value to set, in its native "ov::Any" representation.
     */
    void update(std::string_view key, const ov::Any& value);

    /**
     * @brief Sets the options for which an associated environment variable is defined and exported.
     * Values which cannot be parsed are ignored and only reported as warnings.
     */
    void parseEnvVars();

    /**
     * @brief Checks if a value has been set for the given option.
     * @tparam Opt The option to check.
     * @return True if a value was set, false if only the default value (if any) is available.
     */
    template <class Opt>
    bool has() const;

    /**
     * @brief Checks if a value has been set for the option identified by the given key.
     * @param key The key of the option to check.
     * @return True if a value was set, false if only the default value (if any) is available.
     */
    bool has(std::string_view key) const;

    /**
     * @brief Erases the value set for the given option key. Does nothing if no value was set.
     * @param key The key of the option to erase.
     */
    void remove(std::string_view key);

    /**
     * @brief Removes all compile-time and internal compiler configuration entries.
     * This is used when a compiler type is not explicitly selected and the config must be reset
     * to a runtime-only state.
     */
    void removeCompileTimeConfigs();

    /**
     * @brief Retrieves the value of the given option, falling back to its default value if none was set.
     * @tparam Opt The option to retrieve.
     * @return The option's value.
     * @throws ov::Exception If no value was set and the option has no default value.
     */
    template <class Opt>
    typename Opt::ValueType get() const;

    /**
     * @brief Retrieves the value of the given option as a string, using the same fallback rules as "get".
     * @tparam Opt The option to retrieve.
     * @return The string representation of the option's value.
     */
    template <class Opt>
    typename std::string getString() const;

    /**
     * @brief Restores the configuration from a string previously produced by "toString".
     * @param str A space-separated sequence of KEY="VALUE" entries.
     */
    void fromString(const std::string& str);

    // Returns a string with all config keys which have set values
    std::string toString() const;

    /**
     * @brief Checks if a specific option exists in the configuration's descriptor.
     * @param key The key of the option to check.
     * @return True if the option exists, false otherwise.
     */
    bool hasOpt(std::string_view key) const;

    /**
     * @brief Retrieves the OptionBase concept associated with a specific option. Used to check option details.
     * @param key The key of the option to retrieve.
     * @return The `OptionConcept` object representing the option's details.
     */
    details::OptionConcept getOpt(std::string_view key) const;

    /**
     * @brief Adds or updates an internal configuration value for compiler-specific needs.
     * @param key The key of the internal configuration to add or update.
     * @param value The value to set for the internal configuration.
     */
    void addOrUpdateInternal(std::string key, std::string value);

    /**
     * @brief Checks if an internal compiler configuration exists.
     * @param key The key of the internal configuration to check.
     * @return True if the internal configuration exists, false otherwise.
     */
    bool hasInternal(std::string_view key) const;

    /**
     * @brief Retrieves an internal configuration value by its key.
     * @param key The key of the internal configuration to retrieve.
     * @return The value associated with the specified internal configuration key.
     */
    std::string getInternal(std::string_view key) const;

    /**
     * @brief Generates a compiler configuration string for options supported by the current compiler.
     * @param isSupported Predicate used to filter compile-time and internal compiler options.
     * @return A string containing the supported configuration keys and values.
     */
    std::string toStringForCompiler(const std::function<bool(const std::string&)>& isSupported) const;

private:
    std::shared_ptr<const OptionsDesc> _desc;
    std::unordered_map<std::string_view, std::shared_ptr<details::OptionValue>> _impl;

    ConfigMap _internal_compiler_configs;  ///< Map to store internal (hidden) configurations used for compiler.

    Logger _log{Logger::global().clone("Config")};
};

template <class Opt>
bool Config::has() const {
    return _impl.count(Opt::key().data()) != 0;
}

template <class Opt>
typename Opt::ValueType Config::get() const {
    using ValueType = typename Opt::ValueType;

    _log.trace("Get value for the option '%s'", Opt::key().data());

    const auto it = _impl.find(Opt::key());

    if (it == _impl.end()) {
        const std::optional<ValueType> optional = Opt::defaultValue();
        _log.trace("The option '%s' was not set by user, try default value", Opt::key().data());

        OPENVINO_ASSERT(optional.has_value(),
                        "Option '",
                        Opt::key().data(),
                        "' was not provided, no default value is available");
        return optional.value();
    }

    OPENVINO_ASSERT(it->second != nullptr, "Got NULL OptionValue for :", Opt::key().data());

    const auto optVal = std::dynamic_pointer_cast<details::OptionValueImpl<Opt, ValueType>>(it->second);
#if defined(__CHROMIUMOS__) || defined(__ANDROID__)
    if (optVal == nullptr) {
        if (Opt::getTypeName() == it->second->getTypeName()) {
            const auto val = std::static_pointer_cast<details::OptionValueImpl<Opt, ValueType>>(it->second);
            return val->getValue();
        }
    }
#endif
    OPENVINO_ASSERT(optVal != nullptr,
                    "Option '",
                    Opt::key().data(),
                    "' has wrong parsed type: expected '",
                    Opt::getTypeName().data(),
                    "', got '",
                    it->second->getTypeName().data(),
                    "'");

    return optVal->getValue();
}

template <class Opt>
typename std::string Config::getString() const {
    typename Opt::ValueType value = Config::get<Opt>();

    return Opt::toString(value);
}

}  // namespace intel_npu
