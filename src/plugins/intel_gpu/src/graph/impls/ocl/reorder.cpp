// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reorder/reorder_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reorder_impl : typed_primitive_impl_ocl<reorder> {
    using parent = typed_primitive_impl_ocl<reorder>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reorder_impl>(*this);
    }

    reorder_impl() : parent() {}

    explicit reorder_impl(const reorder_impl& other) : parent(other),
        _can_be_optimized(other._can_be_optimized),
        _has_mean(other._has_mean) {}

    reorder_impl(const reorder_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<reorder>());
        const auto& node = arg.as<reorder>();
        _can_be_optimized = node.can_be_optimized();
        _has_mean = node.has_mean();
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _can_be_optimized;
        ob << _has_mean;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _can_be_optimized;
        ib >> _has_mean;
    }

protected:
    bool optimized_out(reorder_inst& instance) const override {
        return parent::optimized_out(instance) || _can_be_optimized;
    }

    kernel_arguments_data get_arguments(const reorder_inst& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        auto input = &instance.input_memory();
        auto input_layout = input->get_layout();
        if (_has_mean) {
            if (input_layout.format == cldnn::format::nv12) {
                args.bias = instance.mean_nv12_memory();
            } else {
                args.bias = instance.mean_memory();
            }
        }
        return args;
    }

public:
    static primitive_impl* create(const reorder_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto&& output_layout = impl_param.output_layout;
        auto reorder_params = get_default_params<kernel_selector::reorder_params>(impl_param);
        auto reorder_optional_params =
            get_default_optional_params<kernel_selector::reorder_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.inputs_count(); i++) {
            reorder_params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[i]));
        }
        if (impl_param.output_layout.data_padding) {
            reorder_params.has_padded_output = true;
        }

        if (arg.has_mean()) {
            if (impl_param.input_layouts[0].format == cldnn::format::nv12) {
                const auto& mean_layout = arg.mean_nv12().get_output_layout();
                reorder_params.mean = convert_data_tensor(mean_layout);
                reorder_params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            } else {
                const auto mean_idx = 1;
                const auto& mean_layout = impl_param.input_layouts[mean_idx];
                reorder_params.mean = convert_data_tensor(mean_layout);
                reorder_params.mode = kernel_selector::mean_subtruct_mode::IN_BUFFER;
            }
        } else if (prim->subtract_per_feature.empty() == false) {
            reorder_params.mode = kernel_selector::mean_subtruct_mode::INSIDE_PARAMS;
            reorder_params.meanValues = prim->subtract_per_feature;
        } else {
            reorder_params.mode = kernel_selector::mean_subtruct_mode::NONE;
        }

        if (reorder_params.mode != kernel_selector::mean_subtruct_mode::NONE) {
            switch (prim->mean_mode) {
                case reorder_mean_mode::none:
                    reorder_params.mean_op = kernel_selector::mean_op::NONE;
                    break;
                case reorder_mean_mode::mul:
                    reorder_params.mean_op = kernel_selector::mean_op::MUL;
                    break;
                case reorder_mean_mode::subtract:
                    reorder_params.mean_op = kernel_selector::mean_op::SUB;
                    break;
                case reorder_mean_mode::div:
                    reorder_params.mean_op = kernel_selector::mean_op::DIV;
                    break;
                default:
                    throw std::out_of_range(arg.id() + ": unsupported mean_mode value.");
            }
        }

        if (output_layout.format == format::winograd_2x3_s1_data) {
            reorder_params.winograd_input_offset_x = 0;
            reorder_params.winograd_input_offset_y = 0;
            reorder_params.winograd_nr_tiles_x = ceil_div(output_layout.spatial(0), 4);
        }

        reorder_params.winograd = impl_param.input_layouts[0].format.is_winograd() || output_layout.format.is_winograd();

        auto& kernel_selector = kernel_selector::reorder_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(reorder_params, reorder_optional_params);


        auto reorder = new reorder_impl(arg, best_kernel);

        return reorder;
    }

private:
    bool _can_be_optimized;
    bool _has_mean;
};

namespace detail {

attach_reorder_impl::attach_reorder_impl() {
    implementation_map<reorder>::add(impl_types::ocl, reorder_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reorder_impl, cldnn::object_type::REORDER_IMPL)
