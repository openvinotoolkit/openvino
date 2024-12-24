// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "resample_inst.h"
#include "impls/registry/implementation_map.hpp"

#include "openvino/op/interpolate.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <utility>

namespace cldnn {
namespace cpu {

namespace {



}  // namespace

struct resample_impl : public typed_primitive_impl<resample> {
    using parent = typed_primitive_impl<resample>;
    using parent::parent;

    using InterpolateMode = ov::op::v4::Interpolate::InterpolateMode;
    using CoordinateTransformMode = ov::op::v4::Interpolate::CoordinateTransformMode;
    using Nearest_mode = ov::op::v4::Interpolate::NearestMode;
    using InterpolateAttrs = ov::op::v4::Interpolate::InterpolateAttrs;
    using ShapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode;

    std::shared_ptr<ov::op::Op> op;

    std::vector<int64_t> sizes;
    std::vector<float> scales;
    std::vector<int64_t> axes;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    InterpolateMode operation_type = InterpolateMode::LINEAR;
    ShapeCalcMode shape_calc_mode = ShapeCalcMode::SIZES;
    int32_t antialias = 0;
    float cube_coeff = -0.75f;
    CoordinateTransformMode coord_trans_mode = CoordinateTransformMode::HALF_PIXEL;
    Nearest_mode round_mode = Nearest_mode::ROUND_PREFER_FLOOR;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::resample_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<resample_impl>(*this);
    }

    resample_impl() : parent("resample_cpu_impl") {}

    explicit resample_impl(const resample_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<resample>(), "[GPU] Incorrect program_node type");
        const auto& node = arg.as<resample>();
        
        sizes = node.get_primitive()->sizes;
        scales = node.get_primitive()->scales;
        axes = node.get_primitive()->axes;
        
        pads_begin = node.get_primitive()->pads_begin;
        pads_end = node.get_primitive()->pads_end;
        operation_type = node.get_primitive()->operation_type;
        shape_calc_mode = node.get_primitive()->shape_calc_mode;
        antialias = node.get_primitive()->antialias;
        cube_coeff = node.get_primitive()->cube_coeff;
        coord_trans_mode = node.get_primitive()->coord_trans_mode;
        round_mode = node.get_primitive()->round_mode;
    }

    // void save(BinaryOutputBuffer& ob) const override {
    //     parent::save(ob);
    //     ob << make_data(&mode, sizeof(eltwise_mode));
    //     ob << coefficients;
    // }

    // void load(BinaryInputBuffer& ib) override {
    //     parent::load(ib);
    //     ib >> make_data(&mode, sizeof(eltwise_mode));
    //     ib >> coefficients;
    // }

    event::ptr execute_impl(const std::vector<event::ptr>& events, resample_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "resample::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        auto params = instance.get_impl_params();

        // Set input tensors
        ov::TensorVector input_host_tensors;
        auto input_mem_ptr = instance.input_memory_ptr();
        cldnn::mem_lock<uint8_t, mem_lock_type::read> input_lock(input_mem_ptr, stream);
        
        for (size_t i = 0; i < params->input_layouts.size(); i++) {
            auto input_tensor = make_tensor(params->input_layouts[0], input_lock.data());
            input_host_tensors.push_back(input_tensor);
        }

        if (input_host_tensors.size() == 1) {
            auto target_shape_sizes = params->output_layouts[0].get_tensor().sizes();
            std::vector<int64_t> target_shape_ps;
            for (size_t i = 0; i < axes.size(); i++)
                target_shape_ps.push_back(target_shape_sizes[i]);

            auto target_shape_tensor = ov::Tensor(ov::element::i32, {target_shape_ps.size()}, target_shape_ps.data());
            input_host_tensors.push_back(target_shape_tensor);

            if (shape_calc_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SIZES) {
                auto new_scales = scales;
                auto input_shape_sizes = params->input_layouts[0].get_tensor().sizes();
                for (size_t i = 0; i < sizes.size(); i++)
                    new_scales[i] = sizes[i] / input_shape_sizes[i];

                auto scales_tensor = ov::Tensor(ov::element::f32, {new_scales.size()}, new_scales.data());
                input_host_tensors.push_back(scales_tensor);
                shape_calc_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
            } else if (shape_calc_mode == ov::op::util::InterpolateBase::ShapeCalcMode::SCALES) {
                auto scales_tensor = ov::Tensor(ov::element::f32, {scales.size()}, scales.data());
                input_host_tensors.push_back(scales_tensor);
            } else {
                OPENVINO_ASSERT(false, "[GPU] Not supported Interpolate ShapeCalcMode", instance.id());
            }
            
            auto axes_tensor = ov::Tensor(ov::element::i64, {axes.size()}, axes.data());
            input_host_tensors.push_back(axes_tensor);
        }

        // set output tensors
        ov::TensorVector output_host_tensors;
        auto output_mem_ptr = instance.output_memory_ptr();
        cldnn::mem_lock<uint8_t, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        auto output_tensor = make_tensor(params->output_layouts[0], output_lock.data());
        output_host_tensors.push_back(output_tensor);

        // Set Attrs
        InterpolateAttrs attrs;
        attrs.mode                              = operation_type;
        attrs.shape_calculation_mode            = shape_calc_mode;
        attrs.pads_begin                        = pads_begin;
        attrs.pads_end                          = pads_end;
        attrs.coordinate_transformation_mode    = coord_trans_mode;
        attrs.nearest_mode                      = round_mode;
        attrs.antialias                         = antialias;
        attrs.cube_coeff                        = cube_coeff;

        if (!op) {
            auto interp = std::make_shared<ov::op::v4::Interpolate>();
            interp->set_attrs(attrs);
            op = interp;
        }

        OPENVINO_ASSERT(op->evaluate(output_host_tensors, input_host_tensors),
                        "[GPU] Couldn't execute resample primitive with id ", instance.id());

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const resample_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<resample_impl>();
    }
};


namespace detail {

attach_resample_impl::attach_resample_impl() {
    // auto formats = {
    //     format::bfyx,
    // };

    // auto types = {
    //     data_types::f32,
    // };

    // implementation_map<resample>::add(impl_types::cpu, shape_types::static_shape, resample_impl::create, types, formats);
    // implementation_map<resample>::add(impl_types::cpu, shape_types::dynamic_shape, resample_impl::create, types, formats);

    //std::set<implementation_map<resample>::key_type> keys;

    const auto types = {data_types::f32, data_types::i32};
    const auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };
    // for (const auto type : types) {
    //     for (const auto format : formats) {
    //         keys.emplace(type, format);
    //     }
    // }

    // keys.emplace(data_types::f32, format::yxfb);

    implementation_map<resample>::add(impl_types::cpu, shape_types::static_shape, resample_impl::create, types, formats);
    implementation_map<resample>::add(impl_types::cpu, shape_types::dynamic_shape, resample_impl::create, types, formats);
}

}  // namespace detail


}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::resample_impl)
