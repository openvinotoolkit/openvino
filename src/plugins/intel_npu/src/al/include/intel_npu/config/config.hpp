// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
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

#define TYPE_PRINTER(type)                                    \
    template <>                                               \
    struct TypePrinter<type> {                                \
        static constexpr bool hasName() { return true; }      \
        static constexpr const char* name() { return #type; } \
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

//
// OptionParser
//

template <typename T>
struct OptionParser;

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

template <>
struct OptionParser<ov::log::Level> final {
    static ov::log::Level parse(std::string_view val);
};

template <>
struct OptionParser<ov::hint::ExecutionMode> final {
    static ov::hint::ExecutionMode parse(std::string_view val);
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
        std::istringstream stream(val.data());

        Rep count{};
        if (stream >> count) {
            OPENVINO_ASSERT(count >= 0,
                            "Value '",
                            count,
                            "' is not a valid time duration, non-negative values expected");
            return std::chrono::duration<Rep, Period>(count);
        }

        OPENVINO_THROW("Can't parse '", val.data(), "' as time duration");
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
        } else if constexpr (std::is_enum_v<std::decay_t<T>>) {
            ss << stringifyEnum(val);
            return ss.str();
        } else {
            ss << val;
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

template <>
struct OptionPrinter<ov::log::Level> final {
    static std::string toString(ov::log::Level val);
};

template <>
struct OptionPrinter<ov::hint::ExecutionMode> final {
    static std::string toString(ov::hint::ExecutionMode val);
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

    // Overload this for private options.
    static bool isPublic() {
        return true;
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
    bool (*isPublic)() = nullptr;
    std::shared_ptr<OptionValue> (*validateAndParse)(std::string_view val) = nullptr;
};

template <class Opt>
std::shared_ptr<OptionValue> validateAndParse(std::string_view val) {
    using ValueType = typename Opt::ValueType;

    try {
        auto parsedVal = Opt::parse(val);
        Opt::validateValue(parsedVal);
        return std::make_shared<OptionValueImpl<Opt, ValueType>>(std::move(parsedVal), &Opt::toString);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to parse '", Opt::key().data(), "' option : ", e.what());
    }
}

template <class Opt>
OptionConcept makeOptionModel() {
    return {&Opt::key, &Opt::envVar, &Opt::mode, &Opt::isPublic, &validateAndParse<Opt>};
}

}  // namespace details

//
// OptionsDesc
//

class OptionsDesc final {
public:
    template <class Opt>
    void add();

    std::vector<std::string> getSupported(bool includePrivate = false) const;

    details::OptionConcept get(std::string_view key, OptionMode mode) const;
    void walk(std::function<void(const details::OptionConcept&)> cb) const;

private:
    std::unordered_map<std::string, details::OptionConcept> _impl;
    std::unordered_map<std::string, std::string> _deprecated;
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
    using ImplMap = std::unordered_map<std::string, std::shared_ptr<details::OptionValue>>;

    explicit Config(const std::shared_ptr<const OptionsDesc>& desc);

    void update(const ConfigMap& options, OptionMode mode = OptionMode::Both);

    void parseEnvVars();

    template <class Opt>
    bool has() const;

    template <class Opt>
    typename Opt::ValueType get() const;

    template <class Opt>
    typename std::string getString() const;

    std::string toString() const;

private:
    std::shared_ptr<const OptionsDesc> _desc;
    ImplMap _impl;
};

template <class Opt>
bool Config::has() const {
    return _impl.count(Opt::key().data()) != 0;
}

template <class Opt>
typename Opt::ValueType Config::get() const {
    using ValueType = typename Opt::ValueType;

    auto log = Logger::global().clone("Config");
    log.trace("Get value for the option '%s'", Opt::key().data());

    const auto it = _impl.find(Opt::key().data());

    if (it == _impl.end()) {
        const std::optional<ValueType> optional = Opt::defaultValue();
        log.trace("The option '%s' was not set by user, try default value", Opt::key().data());

        OPENVINO_ASSERT(optional.has_value(),
                        "Option '",
                        Opt::key().data(),
                        "' was not provided, no default value is available");
        return optional.value();
    }

    OPENVINO_ASSERT(it->second != nullptr, "Got NULL OptionValue for :", Opt::key().data());

    const auto optVal = std::dynamic_pointer_cast<details::OptionValueImpl<Opt, ValueType>>(it->second);
#if defined(__CHROMIUMOS__)
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

//
// envVarStrToBool
//

bool envVarStrToBool(const char* varName, const char* varValue);

}  // namespace intel_npu
