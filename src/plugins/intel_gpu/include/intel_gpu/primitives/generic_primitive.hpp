// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include <vector>
#include <string>

namespace cldnn {

struct generic_primitive : public primitive_base<generic_primitive> {
    CLDNN_DECLARE_PRIMITIVE(generic_primitive)

    typedef std::function<event::ptr(const std::vector<event::ptr>& dependent_events,
                                        cldnn::stream& stream,
                                        const std::vector<memory::ptr>& inputs,
                                        const std::vector<memory::ptr>& outputs)>
            execute_function;

    typedef std::function<std::vector<ov::PartialShape>(const std::vector<ov::PartialShape>& input_shapes)>
            shape_infer_function;

    generic_primitive() : primitive_base("", {}) {}

    generic_primitive(const primitive_id& id,
                        const std::vector<input_info>& inputs,
                        const execute_function& execute_f,
                        const shape_infer_function& shape_infer_f,
                        size_t num_outputs,
                        const std::vector<optional_data_type>& out_types)
            : primitive_base(id, {inputs}, num_outputs, out_types),
                execute_f(execute_f),
                shape_infer_f(shape_infer_f) {}

    const execute_function execute_f;
    const shape_infer_function shape_infer_f;
};

}  // namespace cldnn
