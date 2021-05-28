// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>
#include "cldnn.hpp"
#include "compounds.h"
#include "primitive.hpp"
#include <vector>
#include <memory>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_topology Network Topology
/// @{

struct topology_impl;

/// @brief Network topology to be defined by user.
struct topology {
    /// @brief Constructs empty network topology.
    topology();

    /// @brief Constructs topology containing primitives provided in argument(s).
    template <class... Args>
    explicit topology(const Args&... args) : topology() {
        add<Args...>(args...);
    }

    /// @brief Copy construction.
    topology(const topology& other) : _impl(other._impl) { retain(); }

    /// @brief Copy assignment.
    topology& operator=(const topology& other) {
        if (_impl == other._impl)
            return *this;
        release();
        _impl = other._impl;
        retain();
        return *this;
    }

    /// Construct C++ topology based on C API @p cldnn_topology
    explicit topology(topology_impl* other) : _impl(other) {
        if (_impl == nullptr)
            throw std::invalid_argument("implementation pointer should not be null");
    }

    /// @brief Releases wrapped C API @ref cldnn_topology.
    ~topology() { release(); }

    friend bool operator==(const topology& lhs, const topology& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const topology& lhs, const topology& rhs) { return !(lhs == rhs); }

    void add_primitive(std::shared_ptr<primitive> desc);

    /// @brief Adds a primitive to topology.
    template <class PType>
    void add(PType const& desc) {
        add_primitive(std::static_pointer_cast<primitive>(std::make_shared<PType>(desc)));
    }

    /// @brief Adds primitives to topology.
    template <class PType, class... Args>
    void add(PType const& desc, Args const&... args) {
        add(desc);
        add<Args...>(args...);
    }

    /// @brief Returns wrapped implementation pointer.
    topology_impl* get() const { return _impl; }

    const std::vector<primitive_id> get_primitive_ids() const;

    void change_input_layout(primitive_id id, const layout& new_layout);

    const std::shared_ptr<primitive>& at(const primitive_id& id) const;

private:
    friend struct engine;
    friend struct network;
    topology_impl* _impl;

    void retain();
    void release();
};

CLDNN_API_CLASS(topology)
/// @}
/// @}
}  // namespace cldnn
