// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>

#include "register.hpp"
#include "crop_inst.h"
#include "implementation_map.hpp"

#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace cpu {

struct crop_impl : public typed_primitive_impl<crop> {
    using parent = typed_primitive_impl<crop>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<crop_impl>(*this);
    }

    crop_impl() : parent("crop_cpu_impl") {}

    explicit crop_impl(const crop_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<crop>());
    }

    template <class T>
    void calculate_crop(const cldnn::kernel_impl_params* params, cldnn::memory::ptr input_mem_ptr, cldnn::memory::ptr output_mem_ptr, cldnn::stream &stream) {
        auto input_layout = params->input_layouts[0];
        auto output_layout = params->output_layouts[0];
        auto input_offset = params->input_offsets[0];

        const auto max_dims_num = 6;
        const auto offsets_shape = input_offset.get_partial_shape(input_layout.get_rank()).to_shape();

        int offsets[max_dims_num] = {0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 2; i++)
            offsets[i] = static_cast<int>(offsets_shape[i]);

        for (size_t i = 2; i < offsets_shape.size(); i++)
            offsets[i + max_dims_num - offsets_shape.size()] = static_cast<int>(offsets_shape[i]);

        cldnn::mem_lock<T, mem_lock_type::read> input_lock(input_mem_ptr, stream);
        cldnn::mem_lock<T, mem_lock_type::write> output_lock(output_mem_ptr, stream);

        auto size_out = output_layout.get_tensor();
        auto padded_output = static_cast<bool>(output_mem_ptr->get_layout().data_padding);

        if (padded_output) {
            for (int b = 0; b < size_out.batch[0]; ++b) {
                for (int f = 0; f < size_out.feature[0]; ++f) {
                    for (int w = 0; w < size_out.spatial[3]; ++w) {
                        for (int z = 0; z < size_out.spatial[2]; ++z) {
                            for (int y = 0; y < size_out.spatial[1]; ++y) {
                                cldnn::tensor input_t(cldnn::group(0),
                                                    cldnn::batch(b + offsets[0]), cldnn::feature(f + offsets[1]),
                                                    cldnn::spatial(offsets[5], y + offsets[4], z + offsets[3], w + offsets[2]));
                                cldnn::tensor output_t(cldnn::group(0),
                                                    cldnn::batch(b), cldnn::feature(f),
                                                    cldnn::spatial(0, y, z, w));
                                size_t input_idx = input_layout.get_linear_offset(input_t);
                                size_t output_idx = output_layout.get_linear_offset(output_t);
                                for (int x = 0; x < offsets[5] + size_out.spatial[0]; ++x) {
                                    output_lock[output_idx++] = input_lock[input_idx++];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            size_t out_idx = 0;
            for (int b = offsets[0]; b < offsets[0] + size_out.batch[0]; ++b) {
                for (int f = offsets[1]; f < offsets[1] + size_out.feature[0]; ++f) {
                    for (int w = offsets[2]; w < offsets[2] + size_out.spatial[3]; ++w) {
                        for (int z = offsets[3]; z < offsets[3] + size_out.spatial[2]; ++z) {
                            for (int y = offsets[4]; y < offsets[4] + size_out.spatial[1]; ++y) {
                                cldnn::tensor input_t(cldnn::group(0), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(offsets[5], y, z, w));
                                size_t input_idx = input_layout.get_linear_offset(input_t);
                                for (int x = offsets[5]; x < offsets[5] + size_out.spatial[0]; ++x) {
                                    output_lock[out_idx++] = input_lock[input_idx++];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, crop_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "crop::execute_impl");
        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        auto ev = stream.create_user_event(false);

        auto params = instance.get_impl_params();

        auto input_mem_ptr = instance.input_memory_ptr();
        auto output_mem_ptr = instance.output_memory_ptr();

        switch (params->input_layouts[0].data_type) {
        case data_types::f32:
            calculate_crop<float>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::f16:
            calculate_crop<half_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i64:
            calculate_crop<int64_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i32:
            calculate_crop<int>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::u8:
            calculate_crop<uint8_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        case data_types::i8:
            calculate_crop<int8_t>(params, input_mem_ptr, output_mem_ptr, stream);
            break;
        default:
            OPENVINO_THROW("[GPU] Couldn't execute crop operation: unsupported input data type");
        }

        ev->set();

        return ev;
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update_dispatch_data(const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const crop_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<crop_impl>();
    }
};


namespace detail {

attach_crop_impl::attach_crop_impl() {
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
        data_types::i32,
        data_types::i64,
    };

    implementation_map<crop>::add(impl_types::cpu, shape_types::static_shape, crop_impl::create, types, formats);
    implementation_map<crop>::add(impl_types::cpu, shape_types::dynamic_shape, crop_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::crop_impl)
