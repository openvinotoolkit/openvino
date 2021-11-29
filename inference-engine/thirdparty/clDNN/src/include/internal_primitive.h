// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "api/primitive.hpp"

#include "primitive_type.h"

namespace cldnn {

struct internal_primitive : public primitive {
public:
    // a helper structure which returns true when compared with any primitive_type which is internal
    struct internal_primitive_generic_type {
        friend bool operator==(internal_primitive_generic_type, primitive_type_id type) {
            return type->is_internal_type();
        }

        friend bool operator==(primitive_type_id type, internal_primitive_generic_type) {
            return type->is_internal_type();
        }

        friend bool operator==(internal_primitive_generic_type, internal_primitive_generic_type) { return true; }
    };

    static internal_primitive_generic_type type_id() { return {}; }

private:
    internal_primitive() = delete;
    internal_primitive(internal_primitive const&) = delete;
    internal_primitive(internal_primitive&&) = delete;
};

}  // namespace cldnn