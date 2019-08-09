/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include "api/CPP/primitive.hpp"

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