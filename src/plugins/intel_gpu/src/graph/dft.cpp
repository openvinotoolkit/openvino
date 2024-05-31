// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_inst.h"
#include "primitive_type_base.h"
#include "fft_base_shape_inference.hpp"
#include "rdft_shape_inference.hpp"
#include "irdft_shape_inference.hpp"

#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dft)

layout dft_inst::calc_output_layout(dft_node const& node, kernel_impl_params const& impl_param) {
    const auto primitive = impl_param.typed_desc<dft>();
    const auto input_layout = impl_param.get_input_layout();

    std::vector<tensor::value_type> dims_converted(primitive->output_shape.size());
    std::transform(primitive->output_shape.begin(),
                   primitive->output_shape.end(),
                   dims_converted.begin(),
                   [](size_t value) {
                       return static_cast<int>(value);
                   });

    // Extend shape to 4d by pushing ones at the end (needed to support less than 4d cases)
    for (auto i = dims_converted.size(); i < 4; ++i) {
        auto it = dims_converted.end();
        // For IRDFT push ones at the end, for other DTFs push ones before the last dim
        if (primitive->direction != dft_direction::inverse || primitive->mode != dft_mode::real) {
            it = std::prev(it);
        }
        dims_converted.insert(it, 1);
    }

    const auto output_format = format::adjust_to_rank(input_layout.format, dims_converted.size());
    return {input_layout.data_type, output_format, tensor(output_format, dims_converted)};
}

template<typename ShapeType>
std::vector<layout> dft_inst::calc_output_layouts(dft_node const& /*node*/, kernel_impl_params const& impl_param) {
    std::vector<layout> layouts;

    const auto primitive = impl_param.typed_desc<dft>();
    const auto& input0_layout = impl_param.get_input_layout(0);
    if (impl_param.input_layouts.size() == 1) {
        auto dt = primitive->get_output_data_type(0).value_or(input0_layout.data_type);
        format output_format = format::adjust_to_rank(input0_layout.format, primitive->output_shape.size());
        layouts.push_back(layout{primitive->output_shape, dt, output_format});
        return layouts;
    }

    const auto& input1_layout = impl_param.get_input_layout(1);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>()
    };

    if (impl_param.input_layouts.size() == 3)
        input_shapes.push_back(impl_param.get_input_layout(2).get<ShapeType>());

    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::unordered_map<size_t, ov::Tensor> const_data;
    ov::Tensor axes_tensor, signal_size_tensor;

    auto& memory_deps = impl_param.memory_deps;

    // Consider axes and signal_size are constant case
    if ((primitive->axes.size() > 0) &&
        ((impl_param.input_layouts.size() == 2) || (primitive->signal_size.size() > 0))) {
        auto axes_ptr = reinterpret_cast<uint8_t*>(const_cast<int64_t*>(primitive->axes.data()));
        axes_tensor = ov::Tensor(ov::element::i64, ov::Shape({primitive->axes.size()}), axes_ptr, {});
        const_data.emplace(1, axes_tensor);

        if (primitive->signal_size.size() > 0) {
            auto signal_size_ptr = reinterpret_cast<uint8_t*>(const_cast<int64_t*>(primitive->signal_size.data()));
            signal_size_tensor = ov::Tensor(ov::element::i64, ov::Shape({primitive->signal_size.size()}), signal_size_ptr, {});
            const_data.emplace(2, signal_size_tensor);
        }
    } else {
        if (memory_deps.count(1)) {
            auto axes_mem = memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> axes_lock(axes_mem, impl_param.get_stream());
            axes_tensor = make_tensor(axes_mem->get_layout(), axes_lock.data());
            const_data.emplace(1, axes_tensor);

            if (memory_deps.count(2)) {
                auto signal_size_mem = memory_deps.at(2);
                cldnn::mem_lock<uint8_t, mem_lock_type::read> signal_size_lock(signal_size_mem, impl_param.get_stream());
                signal_size_tensor = make_tensor(signal_size_mem->get_layout(), signal_size_lock.data());
                const_data.emplace(2, signal_size_tensor);
            }
        }
    }

    const auto tensor_accessor = ov::make_tensor_accessor(const_data);
    if (primitive->mode == cldnn::dft_mode::complex) {
        if (primitive->direction == cldnn::dft_direction::forward) {
            ov::op::v7::DFT op;

            output_shapes = ov::op::shape_infer(&op, input_shapes, tensor_accessor);
        } else {
            ov::op::v7::IDFT op;

            output_shapes = ov::op::shape_infer(&op, input_shapes, tensor_accessor);
        }
    } else {
        if (primitive->direction == cldnn::dft_direction::forward) {
            ov::op::v9::RDFT op;

            output_shapes = ov::op::v9::shape_infer(&op, input_shapes, tensor_accessor);
        } else {
            ov::op::v9::IRDFT op;

            output_shapes = ov::op::v9::shape_infer(&op, input_shapes, tensor_accessor);
        }
    }

    auto dt = primitive->get_output_data_type(0).value_or(impl_param.get_input_layout(0).data_type);
    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());
    layouts.push_back(layout{output_shapes[0], dt, output_format});

    return layouts;
}

template std::vector<layout> dft_inst::calc_output_layouts<ov::PartialShape>(dft_node const& node, kernel_impl_params const& impl_param);

std::string dft_inst::to_string(dft_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite dft_info;
    dft_info.add("axes", desc->axes);
    dft_info.add("signal_size", desc->signal_size);
    dft_info.add("output_shape", desc->output_shape);
    dft_info.add("direction", desc->direction == dft_direction::forward ? "forward" : "inverse");
    dft_info.add("mode", desc->mode == dft_mode::real ? "real" : "complex");

    node_info->add("dft info", dft_info);

    std::ostringstream os;
    node_info->dump(os);
    return os.str();
}

dft_inst::typed_primitive_inst(network& network, dft_node const& node) : parent(network, node) {}
}  // namespace cldnn
