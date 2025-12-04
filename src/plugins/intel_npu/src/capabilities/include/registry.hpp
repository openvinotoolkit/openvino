//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "meta.hpp"
#include "concepts.hpp"

namespace compat {

namespace {

template <class T>
std::unique_ptr<const concepts::Capability> createParser(const uint8_t* bytes) {
    // assert(bytes != nullptr);
    // it is safe and intentional to do reinterpret_cast here
    // as Capability classes binary representation is kept cross-compatible
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return std::make_unique<const concepts::CapabilityModel<T>>(reinterpret_cast<const T*>(bytes));
}

}  // namespace

class Registry {
public:
    static const Registry& get();

    bool isRegistered(uint16_t type) const;

    template <class T>
    bool isRegistered() const {
        static_assert(meta::IsCapability<T>::value, "Type does not satisfy Capability concept");
        return isRegistered(T::ID);
    }

    std::unique_ptr<const concepts::Capability> parseHW(uint16_t type, const uint8_t* bytes) const;
    std::unique_ptr<const concepts::Capability> parseSW(uint16_t type, const uint8_t* bytes) const;

private:
    Registry();
    friend Registry& createRegistry();

    Registry(const Registry&) = delete;
    Registry(Registry&&) = delete;
    Registry& operator=(const Registry&) = delete;
    Registry& operator=(Registry&&) = delete;
    ~Registry() = default;

    using CapabilityCreateFn = std::unique_ptr<const concepts::Capability>(*)(const uint8_t* bytes);
    std::unordered_map<uint16_t, CapabilityCreateFn> _hwCapabilities;
    std::unordered_map<uint16_t, CapabilityCreateFn> _swCapabilities;

    template <class T>
    static void registerCapability(std::unordered_map<uint16_t, CapabilityCreateFn>& capabilities) {
        static_assert(meta::IsCapability<T>::value, "Type does not satisfy Capability concept");

        const auto iteratorAndStatus = capabilities.emplace(T::ID, compat::createParser<T>);
        assert(std::get<1>(iteratorAndStatus));
    }
};

}  // namespace compat
