// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cctype>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "selective_build.h"

namespace openvino {
namespace cc {

template <typename Key, typename T>
class Factory;

template <typename Key, typename T, typename... Args>
class Factory<Key, T(Args...)> {
    Factory(Factory const&) = delete;
    Factory& operator=(Factory const&) = delete;

public:
    using builder_t = std::function<T(Args...)>;

    Factory(const std::string& name) : name(name) {}

#ifdef SELECTIVE_BUILD
#    define registerNodeIfRequired(Module, Name, key, Impl) \
        OV_PP_EXPAND(OV_PP_CAT(registerImpl, OV_CC_SCOPE_IS_ENABLED(OV_PP_CAT3(Module, _, Name))) < Impl > (key))
#    define createNodeIfRegistered(Module, key, ...) createImpl(key, __VA_ARGS__)

    template <typename Impl>
    void registerImpl0(const Key&) {}

    template <typename Impl>
    void registerImpl1(const Key& key) {
        builders[key] = [](Args... args) -> T {
            Impl* impl = new Impl(std::move(args)...);
            return static_cast<T>(impl);
        };
    }

    T createImpl(const Key& key, Args... args) {
        auto builder = builders.find(key);
        if (builder != builders.end()) {
            return builder->second(std::move(args)...);
        }
        return nullptr;
    }

#elif defined(SELECTIVE_BUILD_ANALYZER)
#    define registerNodeIfRequired(Module, Name, key, Impl) \
        registerImpl<OV_PP_CAT(FACTORY_, Module), Impl>(key, OV_PP_TOSTRING(Name))
#    define createNodeIfRegistered(Module, key, ...) createImpl<OV_PP_CAT(FACTORY_, Module)>(key, __VA_ARGS__)

    template <openvino::itt::domain_t (*domain)(), typename Impl>
    void registerImpl(const Key& key, const char* typeName) {
        validate_type(typeName);
        const std::string task_name = "REG$" + name + "$" + to_string(key) + "$" + typeName;
        openvino::itt::ScopedTask<domain> task(openvino::itt::handle(task_name));
        builders[key] = [](Args... args) -> T {
            Impl* impl = new Impl(std::move(args)...);
            return static_cast<T>(impl);
        };
    }

    template <openvino::itt::domain_t (*domain)()>
    T createImpl(const Key& key, Args... args) {
        auto builder = builders.find(key);
        if (builder != builders.end()) {
            const std::string task_name = "CREATE$" + name + "$" + to_string(key);
            openvino::itt::ScopedTask<domain> task(openvino::itt::handle(task_name));
            return builder->second(std::move(args)...);
        }
        return nullptr;
    }

#else

#    define registerNodeIfRequired(Module, Name, key, Impl) registerImpl<Impl>(key)
#    define createNodeIfRegistered(Module, key, ...)        createImpl(key, __VA_ARGS__)

    template <typename Impl>
    void registerImpl(const Key& key) {
        builders[key] = [](Args... args) -> T {
            Impl* impl = new Impl(std::move(args)...);
            return static_cast<T>(impl);
        };
    }

    T createImpl(const Key& key, Args... args) {
        auto builder = builders.find(key);
        if (builder != builders.end()) {
            return builder->second(std::move(args)...);
        }
        return nullptr;
    }
#endif

    template <typename Fn>
    void foreach (Fn fn) const {
        for (auto itm : builders)
            fn(itm);
    }

    size_t size() const noexcept {
        return builders.size();
    }

private:
    void validate_type(const char* name) const {
        for (const char* ch = name; *ch; ++ch) {
            if (!std::isalnum(*ch) && *ch != '_')
                throw std::runtime_error(std::string("Invalid identifier name: '") + name +
                                         "'. Allowed characters should be alphanumeric or '_'");
        }
    }

    const std::string& to_string(const std::string& str) const noexcept {
        return str;
    }

    template <typename V, typename std::enable_if<std::is_enum<V>::value, bool>::type = true>
    std::string to_string(V val) const {
        return std::to_string(static_cast<int>(val));
    }

    template <typename V, typename std::enable_if<!std::is_enum<V>::value, bool>::type = true>
    std::string to_string(V val) const {
        return std::to_string(val);
    }

    template <typename K>
    struct EnumClassHash {
        std::size_t operator()(K t) const {
            return static_cast<std::size_t>(t);
        }
    };

    using hash_t = typename std::conditional<std::is_enum<Key>::value, EnumClassHash<Key>, std::hash<Key>>::type;
    using map_t = std::unordered_map<Key, builder_t, hash_t>;

    const std::string name;
    map_t builders;
};

}  // namespace cc
}  // namespace openvino
