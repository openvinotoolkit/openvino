// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "kernel_base.h"
#include "fully_connected_inst.h"
#include "fully_connected/fully_connected_kernel_selector.h"
#include "fully_connected/fully_connected_params.h"
#include "to_string_utils.h"

namespace cldnn {
namespace ocl {
float convert_element(ov::float16 h);
float convert_element(ov::float16 h) { return static_cast<float>(h); }
struct fully_connected_impl : typed_primitive_impl_ocl<fully_connected> {
    using parent = typed_primitive_impl_ocl<fully_connected>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::fully_connected_kernel_selector;
    using kernel_params_t = kernel_selector::fully_connected_params;
    static int infer_count;
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_impl)

    fully_connected_impl() = default;

    fully_connected_impl(const kernel_selector::kernel_data& kd) {
        const auto& params = kd.weightsReorderParams;
        if (params.is_initialized) {
            // Assumption that kernel data contains already reshaped 2d weights
            auto crop_to_2d = [](const ov::PartialShape& shape) {
                return ov::PartialShape({shape[0], shape[1]});
            };
            std::cout << "in fc ocl get weight reorder" << std::endl;
            std::cout << from_weights_tensor(params.src).to_short_string() << std::endl;
            std::cout << from_weights_tensor(params.dest).to_short_string() << std::endl;
            auto weights_reorder_params = std::make_shared<WeightsReorderParams>(from_weights_tensor(params.src),
                                                                                 from_weights_tensor(params.dest),
                                                                                 params.rotate);
            auto output_layout = weights_reorder_params->get_output_layout();
            output_layout.set_partial_shape(crop_to_2d(output_layout.get_partial_shape()));
            weights_reorder_params->set_output_layout(output_layout);

            _weights_reorder_params = weights_reorder_params;
        }
        _kernel_data = kd;
        _kernel_name = kd.kernelName;
        can_reuse_memory = _kernel_data.can_reuse_memory;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }
    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<fully_connected>& instance) override {
        std::cout << "bell execute TP fully connected!" << std::endl;
        infer_count++;
        stream& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return aggregate_events(events, stream, false, instance.is_output());
        }
        stream.finish(); // extra finish for input copy, to be optimized
        instance.fill_placeholder();
        std::ofstream file_stream("bell_fc_weight_iter" + std::to_string(infer_count) +
                                std::to_string(instance.get_node().as<fully_connected>().w_rank) + ".txt");
        auto input_mem = instance.weights_memory();
        auto&& size = input_mem->get_layout().get_tensor();

        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                        << ", original format: " << cldnn::fmt_to_str(input_mem->get_layout().format) << ")" << std::endl;

        mem_lock<ov::float16, mem_lock_type::read> lock(input_mem, stream);
        auto mem_ptr = lock.data();
        std::stringstream buffer;

        {
            for (size_t i = 0; i < lock.size(); ++i) {
                buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
            }
        }
        file_stream << buffer.str();
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        OPENVINO_ASSERT(_kernels.size() == _kernel_data.kernels.size(), "[GPU] Mismatch between compiled kernels count and expected kernels data\n",
                                                                        "[GPU] Compiled kernels count: ", _kernels.size(), "\n",
                                                                        "[GPU] KernelData count: ", _kernel_data.kernels.size(), "\n",
                                                                        "[GPU] Likely some issue with empty tensor handling happened");
        for (size_t kd_idx = 0; kd_idx < _kernel_data.kernels.size(); ++kd_idx) {
            if (_kernel_data.kernels[kd_idx].skip_execution)
                continue;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernel_data.kernels[kd_idx].params;
            auto args = get_arguments(instance);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue kernel " << kd_idx << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[kd_idx], params, args, tmp_events, needs_completion_event);
            if (_kernel_data.needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return aggregate_events(tmp_events, stream);

        bool group_events = (all_events.size() > 1);
        if (getenv("ENABLE_CCL")) {
            stream.finish(); // can be replaced with need_completion_event?
            auto output_memory_ptr = instance.output_memory_ptr();
            //auto actual_mem = output_memory.get_engine()->reinterpret_buffer(output_memory, output_memory.get_layout());
            //mem_lock<char, mem_lock_type::read_write> lock(actual_mem, stream);
            auto send_ptr = output_memory_ptr->buffer_ptr();
            std::cout << output_memory_ptr->get_allocation_type() << std::endl;
            std::cout << output_memory_ptr->count() << std::endl;
            std::cout << output_memory_ptr->get_layout().to_string() << std::endl;
            std::cout << output_memory_ptr->size() << std::endl;
            std::cout << "bell debug!!!!" << send_ptr << std::endl;
            //auto prec = output.();
            std::cout << "&&&&&&&&" << std::endl;
            std::ofstream file_stream("bell_fc_output_iter_" + std::to_string(infer_count) + "_rank_"
                                    + std::to_string(instance.get_node().as<fully_connected>().w_rank) + ".txt");
            auto&& size = output_memory_ptr->get_layout().get_tensor();

            file_stream << "shape: " << size.to_string() << " ";
            file_stream << "(count: " << size.count()
                            << ", original format: " << cldnn::fmt_to_str(output_memory_ptr->get_layout().format) << ")" << std::endl;

            mem_lock<ov::float16, mem_lock_type::read> lock(instance.output_memory_ptr(), stream);
            auto mem_ptr = lock.data();
            std::stringstream buffer;

            {
                for (size_t i = 0; i < lock.size(); ++i) {
                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
                }
            }
            file_stream << buffer.str();
            Messenger::getInstance().helperAllreducef16(send_ptr, send_ptr, output_memory_ptr->size());
            std::cout << "&&&&&&&&" << std::endl;
            //output_memory.copy_from(stream, *output_host);
        } else {
            std::cout << "bell debug only map output buffer for reference" << std::endl;
            stream.finish(); // can be replaced with need_completion_event?
            auto output_memory_ptr = instance.output_memory_ptr();
            //auto actual_mem = output_memory.get_engine()->reinterpret_buffer(output_memory, output_memory.get_layout());
            //mem_lock<char, mem_lock_type::read_write> lock(actual_mem, stream);
            auto send_ptr = output_memory_ptr->buffer_ptr();
            std::cout << output_memory_ptr->get_allocation_type() << std::endl;
            std::cout << output_memory_ptr->count() << std::endl;
            std::cout << output_memory_ptr->get_layout().to_string() << std::endl;
            std::cout << output_memory_ptr->size() << std::endl;
            std::cout << "bell debug!!!!" << send_ptr << std::endl;
            //auto prec = output.();
            std::cout << "&&&&&&&&" << std::endl;
            std::ofstream file_stream("bell_fc_output_iter" + std::to_string(infer_count) + ".txt");
            auto&& size = output_memory_ptr->get_layout().get_tensor();

            file_stream << "shape: " << size.to_string() << " ";
            file_stream << "(count: " << size.count()
                            << ", original format: " << cldnn::fmt_to_str(output_memory_ptr->get_layout().format) << ")" << std::endl;

            mem_lock<ov::float16, mem_lock_type::read> lock(instance.output_memory_ptr(), stream);
            auto mem_ptr = lock.data();
            std::stringstream buffer;

            {
                for (size_t i = 0; i < lock.size(); ++i) {
                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
                }
            }
            file_stream << buffer.str();
        }
        return aggregate_events(all_events, stream, group_events);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<fully_connected>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<fully_connected>();

        args.weights = instance.weights_memory();
        std::cout << "bell check weight layout!! " << args.weights->get_layout().to_short_string() << std::endl;
        args.bias = instance.bias_term() ? instance.bias_memory() : nullptr;
        args.inputs = {(instance.get_input_rank_placeholder())};
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            args.inputs.push_back(instance.dep_memory_ptr(in_id));

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        std::cout << "check how many times get_kernel_params are called!!!" << std::endl;
        const auto& primitive = impl_param.typed_desc<fully_connected>();

        auto get_fc_input_layouts = [primitive](const std::vector<layout>& input_layouts, bool allow_new_shape_infer) {
            auto reshape_to_2d = [](const ov::PartialShape& shape, const ov::Dimension& feature, size_t rank) {
                if (shape.is_static()) {
                    auto static_shape = shape.to_shape();
                    size_t total = std::accumulate(static_shape.begin(), static_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
                    auto dim = feature.is_static() ? feature.get_length() : static_cast<int64_t>(static_shape[rank - 1]);
                    return ov::PartialShape{ static_cast<int64_t>(total) / dim, dim };
                } else {
                    return ov::PartialShape{ ov::Dimension::dynamic(), feature };
                }
            };

            auto input0_layout = input_layouts[0];
            auto input1_layout = input_layouts[1];
            std::cout << "bell try" << input_layouts[1].to_short_string() << std::endl;

            auto input0_pshape = input0_layout.get_partial_shape();
            auto input1_pshape = input1_layout.get_partial_shape();

            ov::Dimension feature = input0_pshape[std::min(primitive->input_size, static_cast<size_t>(4)) - 1ul];
            if (allow_new_shape_infer) {
                feature = input0_pshape[primitive->input_size - 1ul];
            }

            // TO DO, to remove WA
            if (primitive->input_size > 3) {
                input0_layout.set_partial_shape(reshape_to_2d(input0_pshape, feature, primitive->input_size));
                input0_layout.format = format::bfyx;
            }
            if (input1_pshape.size() != 2) {
                input1_layout.set_partial_shape(reshape_to_2d(input1_pshape, feature, primitive->weights_rank));
                // input1_layout.format = format::bfyx;
            }
            std::cout << "bell try" << input1_layout.to_short_string() << std::endl;
            std::vector<layout> layouts{input0_layout, input1_layout};

            bool has_zp = !primitive->decompression_zero_point.empty();
            bool has_scale = !primitive->decompression_scale.empty();

            size_t offset = primitive->bias.empty() ? 2 : 3;
            if (has_scale) {
                auto scale_layout = input_layouts[offset++];
                layouts.push_back(scale_layout);
            }

            if (has_zp) {
                auto zp_layout = input_layouts[offset];
                layouts.push_back(zp_layout);
            }

            return layouts;
        };

        auto get_fc_output_layout = [primitive](const std::vector<layout>& input_layouts, const layout& output_layout) {
            auto updated_out_layout = output_layout;

            auto input0_pshape = input_layouts[0].get_partial_shape();
            auto input1_pshape = input_layouts[1].get_partial_shape();
            ov::PartialShape updated_out_pshape {input0_pshape[0], input1_pshape[0]};

            if (primitive->input_size == 3) {
                updated_out_pshape = { input0_pshape[0], input0_pshape[1], input1_pshape[0] };
            }
            updated_out_layout.set_partial_shape(updated_out_pshape);

            return updated_out_layout;
        };

        bool allow_new_shape_infer = impl_param.get_program().is_new_shape_infer();
        auto updated_impl_param = impl_param;

        const auto input_layouts = get_fc_input_layouts(impl_param.input_layouts, allow_new_shape_infer);
        updated_impl_param.input_layouts[0] = input_layouts[0];
        updated_impl_param.input_layouts[1] = input_layouts[1];
        updated_impl_param.weights_layout = input_layouts[1];

        std::cout << updated_impl_param.input_layouts[0].to_short_string() << std::endl;
        std::cout << updated_impl_param.input_layouts[1].to_short_string() << std::endl;

        updated_impl_param.output_layouts[0] = get_fc_output_layout(input_layouts, impl_param.get_output_layout());
        std::cout << updated_impl_param.output_layouts[0].to_short_string() << std::endl;
        auto params = get_weights_bias_default_params<kernel_selector::fully_connected_params>(updated_impl_param, false, is_shape_agnostic);
        params.allowInputReordering = true;

        bool commpressed = !primitive->decompression_scale.empty();
        bool with_zp = !primitive->decompression_zero_point.empty();
        if (commpressed) {
            params.compressed = true;
            params.decompression_scale = convert_data_tensor(input_layouts[2]);
            if (with_zp) {
                params.has_decompression_zp = true;
                params.decompression_zero_point = convert_data_tensor(input_layouts[3]);
            } else if (primitive->decompression_zero_point_scalar.has_value()) {
                params.has_decompression_zp = true;
                params.scalar_zp = true;
                params.zp_value = primitive->decompression_zero_point_scalar.value();
            }
        }

        if (primitive->input_size != 3)
            params.outputs = { params.outputs[0].FlattenFeatureAndSpatials() };

        bool is_quantized = true;
        for (auto& input : impl_param.input_layouts)
            is_quantized &= data_type_traits::is_quantized(input.data_type);

        if (is_quantized) {
            params.quantization = kernel_selector::QuantizationType::SYMMETRIC;
        } else {
            params.quantization = kernel_selector::QuantizationType::NONE;
        }

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }

    static bool update_weight_flag;
};
int fully_connected_impl::infer_count = 0;
bool fully_connected_impl::update_weight_flag = false;
namespace detail {

attach_fully_connected_impl::attach_fully_connected_impl() {
    implementation_map<fully_connected>::add(impl_types::ocl,
                                             shape_types::dynamic_shape,
                                             typed_primitive_impl_ocl<fully_connected>::create<fully_connected_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
    });
    implementation_map<fully_connected>::add(impl_types::ocl,
                                             shape_types::static_shape,
                                             typed_primitive_impl_ocl<fully_connected>::create<fully_connected_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_fsv8_bsv8),
        std::make_tuple(data_types::f16, format::bs_fs_fsv8_bsv8),
        std::make_tuple(data_types::f16, format::bs_fs_fsv8_bsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::fully_connected_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::fully_connected)
