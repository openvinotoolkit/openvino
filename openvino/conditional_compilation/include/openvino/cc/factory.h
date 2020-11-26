//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once
#include "selective_build.h"
#include <string>
#include <functional>
#include <type_traits>
#include <unordered_map>

namespace openvino
{
    namespace cc
    {
        template<typename Key, typename T>
        class Factory;

        template<typename Key, typename T, typename ...Args>
        class Factory<Key, T(Args...)> {
            Factory(Factory const&) = delete;
            Factory& operator=(Factory const&) = delete;

        public:
            using builder_t = std::function<T(Args...)>;

            Factory(const std::string & name)
                : name(name) {}

        #ifdef SELECTIVE_BUILD
            #define registerNodeIfRequired(Module, Name, key, Impl)       \
                OV_CC_EXPAND(OV_CC_CAT(registerImpl, OV_CC_SCOPE_IS_ENABLED(OV_CC_CAT3(Module, _, Name)))<Impl>(key))
            #define createNodeIfRegistered(Module, key, ...) createImpl(key, __VA_ARGS__)

            template<typename Impl>
            void registerImpl0(const Key &) {
            }

            template<typename Impl>
            void registerImpl1(const Key & key) {
                builders[key] = [](Args... args) -> T {
                    Impl *impl = new Impl(args...);
                    return static_cast<T>(impl);
                };
            }

            T createImpl(const Key & key, Args... args) {
                auto builder = builders.find(key);
                if (builder != builders.end()) {
                    return builder->second(args...);
                }
                return nullptr;
            }
        #elif defined(SELECTIVE_BUILD_ANALYZER)
            #define registerNodeIfRequired(Module, Name, key, Impl) registerImpl<OV_CC_CAT(FACTORY_, Module), Impl>(key, OV_CC_TOSTRING(Name))
            #define createNodeIfRegistered(Module, key, ...) createImpl<OV_CC_CAT(FACTORY_, Module)>(key, __VA_ARGS__)

            template<openvino::itt::domain_t(*domain)(), typename Impl>
            void registerImpl(const Key & key, const char *typeName) {
                const std::string task_name = "REG$" + name + "$" + to_string(key) + "$" + typeName;
                openvino::itt::ScopedTask<domain> task(openvino::itt::handle(task_name));
                builders[key] = [](Args... args) -> T {
                    Impl *impl = new Impl(args...);
                    return static_cast<T>(impl);
                };
            }

            template<openvino::itt::domain_t(*domain)()>
            T createImpl(const Key & key, Args... args) {
                auto builder = builders.find(key);
                if (builder != builders.end()) {
                    const std::string task_name = "CREATE$" + name + "$" + to_string(key);
                    openvino::itt::ScopedTask<domain> task(openvino::itt::handle(task_name));
                    return builder->second(args...);
                }
                return nullptr;
            }
        #else
            #define registerNodeIfRequired(Module, Name, key, Impl) registerImpl<Impl>(key)
            #define createNodeIfRegistered(Module, key, ...) createImpl(key, __VA_ARGS__)

            template<typename Impl>
            void registerImpl(const Key & key) {
                builders[key] = [](Args... args) -> T {
                    Impl *impl = new Impl(args...);
                    return static_cast<T>(impl);
                };
            }

            T createImpl(const Key & key, Args... args) {
                auto builder = builders.find(key);
                if (builder != builders.end()) {
                    return builder->second(args...);
                }
                return nullptr;
            }
        #endif

            template<typename Fn>
            void foreach(Fn fn) const {
                for (auto itm : builders)
                    fn(itm);
            }

            size_t size() const noexcept {
                return builders.size();
            }

        private:
            const std::string & to_string(const std::string & str) const noexcept {
                return str;
            }

            template<typename V,
                    typename std::enable_if<std::is_enum<V>::value, bool>::type = true>
            std::string to_string(V val) const {
                return std::to_string(static_cast<int>(val));
            }

            template<typename V,
                    typename std::enable_if<!std::is_enum<V>::value, bool>::type = true>
            std::string to_string(V val) const {
                return std::to_string(val);
            }

            using map_t = std::unordered_map<Key, builder_t>;

            const std::string name;
            map_t builders;
        };
    }
}
