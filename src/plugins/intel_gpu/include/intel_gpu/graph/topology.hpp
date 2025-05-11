// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

#include <map>
#include <memory>
#include <vector>

namespace cldnn {

typedef std::map<primitive_id, std::shared_ptr<primitive>> topology_map;

struct topology {
public:
    using ptr = std::shared_ptr<topology>;
    explicit topology(const topology_map& map) : _primitives(map) {}
    topology() : _primitives({}) {}

    /// @brief Constructs topology containing primitives provided in argument(s).
    template <class... Args>
    explicit topology(const Args&... args) : topology() {
        add<Args...>(args...);
    }

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

    const std::shared_ptr<primitive>& at(primitive_id id) const;

    void change_input_layout(const primitive_id& id, const layout& new_layout);

    const topology_map& get_primitives() const { return _primitives; }

    const std::vector<primitive_id> get_primitives_ids() const;

private:
    topology_map _primitives;
};
}  // namespace cldnn
