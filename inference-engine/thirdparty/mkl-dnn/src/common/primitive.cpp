/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "primitive.hpp"
#include "engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::primitive_kind;

status_t mkldnn_primitive_desc_destroy(primitive_desc_t *primitive_desc) {
    if (primitive_desc) delete primitive_desc;
    return success;
}

status_t mkldnn_primitive_create(primitive_t **primitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t *inputs,
        const primitive_t **outputs) {
    if (utils::any_null(primitive, primitive_desc))
        return invalid_arguments;
    for (int i = 0; i < primitive_desc->n_inputs(); ++i) {
        const auto i_p = inputs[i].primitive;
        const auto i_oi = (int)inputs[i].output_index;
        const bool ok = true
            && i_p != nullptr
            && IMPLICATION(i_p->kind() == memory, i_oi == 0)
            && IMPLICATION(i_p->kind() != memory,
                    i_oi < i_p->pd()->n_outputs());
        if (!ok)
            return invalid_arguments;
    }
    for (int i = 0; i < primitive_desc->n_outputs(); ++i)
        if (outputs[i] == nullptr) return invalid_arguments;
    return primitive_desc->create_primitive(primitive, inputs, outputs);
}

status_t mkldnn_primitive_get_primitive_desc(const primitive_t *primitive,
        const primitive_desc_t **primitive_desc) {
    if (utils::any_null(primitive, primitive_desc))
        return invalid_arguments;
    return safe_ptr_assign<const primitive_desc_t>(*primitive_desc,
            primitive->pd());
}

status_t mkldnn_primitive_get_input_at(const primitive_t *primitive,
        size_t index, primitive_at_t *input) {
    if (utils::any_null(primitive, input)
            || index >= primitive->inputs().size())
        return invalid_arguments;
    *input = primitive->inputs()[index];
    return success;
}

status_t mkldnn_primitive_get_output(const primitive_t *primitive,
        size_t index, const primitive_t **output) {
    if (utils::any_null(primitive, output)
            || index >= primitive->outputs().size())
        return invalid_arguments;
    *output = primitive->outputs()[index];
    return success;
}

status_t mkldnn_primitive_destroy(primitive_t *primitive) {
    if (primitive != nullptr)
        delete primitive;
    return success;
}

primitive_at_t mkldnn_primitive_at(const primitive_t *primitive,
        size_t output_index) {
    primitive_at_t result = {primitive, output_index};
    return result;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
