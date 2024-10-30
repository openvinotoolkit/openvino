// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impl_example.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/primitives/reorder.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/sycl_engine.hpp"
#include "ocl/sycl_stream.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive_sycl_base.h"
#include "impls/registry/implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include "sycl/sycl.hpp"
#include "sycl/ext/oneapi/experimental/builtins.hpp"

#include <memory>

#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif

namespace cldnn {
namespace sycl {

template <typename A, typename B>
struct AccumulatorType {
    using type = float;
};

template<> struct AccumulatorType<::sycl::half, ::sycl::half> {
    using type = ::sycl::half;
};

template<> struct AccumulatorType<::sycl::half, uint8_t> {
    using type = ::sycl::half;
};


template<> struct AccumulatorType<::sycl::half, int8_t> {
    using type = ::sycl::half;
};

template<typename AType, typename WType, typename ZPType, typename ScaleType, typename DType>
::sycl::event run_fc_int4_woq(::sycl::queue& queue, bool enqueue_barrier, const AType* a, const WType* w, const ZPType* zp, const ScaleType* s, DType* dst,
                              size_t M, size_t N, size_t K, size_t group_size, size_t groups_num, const ov::Shape& out_shape, optional_value<float> dzp_s) {
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    bool has_value = dzp_s.has_value();
    float dzp_value = dzp_s.value_or(0.0f);
    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint32_t b = index[0];
            const uint32_t m = index[1];
            const uint32_t n = index[2];
            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            const uint32_t dst_index = n + m*N + b*N*M;
            for (uint32_t y = 0; y < K; ++y) {
                const uint32_t input0_offset = y + m*K + b*M*K;
                const uint32_t decomp_offset = (y / group_size % groups_num)*N + n % N;
                const uint32_t filter_offset = y + n*K;
                const uint32_t zp_offset = 0;


                accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                accum_t scale = s[decomp_offset];
                const WType packed = w[filter_offset / 2];

                const WType v0 = packed & 0x0F;
                const WType v1 = (packed & 0xF0) >> 4;
                accum_t unpacked = filter_offset % 2 == 0 ? v0 : v1;

                accum_t filter_val = (unpacked - zp_val) * scale;
                accumulator += a[input0_offset] * filter_val;
            }
            dst[dst_index] = accumulator;
        });
    });
}

template<typename AType, typename WType, typename ZPType, typename ScaleType, typename DType>
::sycl::event run_fc_int8_woq(::sycl::queue& queue, bool enqueue_barrier, const AType* a, const WType* w, const ZPType* zp, const ScaleType* s, DType* dst,
                     size_t M, size_t N, size_t K, size_t group_size, size_t groups_num, const ov::Shape& out_shape, optional_value<float> dzp_s) {
    if (enqueue_barrier) {
        queue.submit([=](::sycl::handler& cgh) {
            cgh.ext_oneapi_barrier();
        });
    }

    bool has_value = dzp_s.has_value();
    float dzp_value = dzp_s.value_or(0.0f);

    return queue.submit([=](::sycl::handler& cgh) {
        cgh.parallel_for(::sycl::range<3>(out_shape[0], out_shape[1], out_shape[2]), [=](::sycl::id<3> index) {
            const uint32_t b = index[0];
            const uint32_t m = index[1];
            const uint32_t n = index[2];
            using accum_t = typename AccumulatorType<AType, WType>::type;
            accum_t accumulator = 0.0f;

            for (uint32_t y = 0; y < K; ++y) {
                const uint32_t input0_offset = y + m*K + b*M*K;
                const uint32_t zp_offset = (y / group_size % groups_num)*N + n % N;
                const uint32_t decomp_offset = (y / group_size % groups_num)*N + n % N;
                const uint32_t filter_offset = y + n*K;

                accum_t zp_val = has_value ? static_cast<accum_t>(dzp_value) : static_cast<accum_t>(zp[zp_offset]);
                accum_t scale = s[decomp_offset];
                accum_t filter_compressed = static_cast<accum_t>(w[filter_offset]);
                accum_t filter_val = (filter_compressed - zp_val) * scale;
                accumulator += a[input0_offset] * filter_val;
            }
            const uint32_t dst_index = n + m*N + b*N*M;
            dst[dst_index] = accumulator;
        });
    });
}

struct fully_connected_sycl_example : typed_primitive_sycl_impl<fully_connected> {
    using parent = typed_primitive_sycl_impl<fully_connected>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::fully_connected_sycl_example)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<fully_connected_sycl_example>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */, typed_primitive_inst<fully_connected>& instance) override {
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<fully_connected>();

        auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output = instance.output_memory_ptr(0);
        auto weights = instance.weights_memory();
        auto bias = instance.bias_term() ? instance.bias_memory() : nullptr;

        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0) };
        size_t in_id = instance.bias_term() ? 3 : 2;
        if (!desc->decompression_scale.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id++));

        if (!desc->decompression_zero_point.empty())
            inputs.push_back(instance.dep_memory_ptr(in_id));

        OPENVINO_ASSERT(!instance.bias_term() && !instance.get_node().has_fused_primitives());

        ov::element::Type_t in_t = params->input_layouts[0].data_type;
        ov::element::Type_t wei_t = params->weights_layout.value().data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        ov::element::Type_t ds_t = params->input_layouts[2].data_type;
        ov::element::Type_t dzp_t = inputs.size() == 3 ? params->input_layouts[3].data_type : ov::element::Type_t::undefined;

        OPENVINO_ASSERT(out_shape.size() == 3);
        size_t M = out_shape[1];
        size_t N = out_shape[2];
        size_t K = params->weights_layout.value().get_partial_shape()[1].get_length();
        size_t groups_num = params->input_layouts[2].get_shape()[1];
        size_t group_size = K / groups_num;

        OPENVINO_ASSERT(inputs.size() >= 2);

        auto dzp_scalar = desc->decompression_zero_point_scalar;

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;

        #define CASE(InputType, WeightsType, ZPType, ScaleType, DstType) \
            in_t == ov::element::InputType && \
            wei_t == ov::element::WeightsType && \
            out_t == ov::element::DstType && \
            ds_t == ov::element::ScaleType && \
            dzp_t == ov::element::ZPType

        if ((CASE(f32, u4, f32, f32, f32)) || (CASE(f32, u4, undefined, f32, f32))) {
            const float* in = static_cast<const float*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());
            const float* ds = static_cast<const float*>(inputs[1]->buffer_ptr());
            const float* dzp = inputs.size() == 3 ? static_cast<const float*>(inputs[2]->buffer_ptr()) : nullptr;

            return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else if ((CASE(f16, u4, f16, f16, f16)) || (CASE(f16, u4, undefined, f16, f16))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
            const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
            const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;


            return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else if ((CASE(f16, u4, f16, f16, f32)) || (CASE(f16, u4, undefined, f16, f32))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());
            const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
            const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;


            return to_ocl_event(stream, run_fc_int4_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else if ((CASE(f32, u8, f32, f32, f32)) || (CASE(f32, u8, undefined, f32, f32))) {
            const float* in = static_cast<const float*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());
            const float* ds = static_cast<const float*>(inputs[1]->buffer_ptr());
            const float* dzp = inputs.size() == 3 ? static_cast<const float*>(inputs[2]->buffer_ptr()) : nullptr;

            return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else if ((CASE(f16, u8, f16, f16, f16)) || (CASE(f16, u8, undefined, f16, f16))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            ::sycl::half* out = static_cast<::sycl::half*>(output->buffer_ptr());
            const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
            const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;

            return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else if ((CASE(f16, u8, f16, f16, f32)) || (CASE(f16, u8, undefined, f16, f32))) {
            const ::sycl::half* in = static_cast<const ::sycl::half*>(inputs[0]->buffer_ptr());
            const uint8_t* wei = static_cast<const uint8_t*>(weights->buffer_ptr());
            float* out = static_cast<float*>(output->buffer_ptr());
            const ::sycl::half* ds = static_cast<const ::sycl::half*>(inputs[1]->buffer_ptr());
            const ::sycl::half* dzp = inputs.size() == 3 ? static_cast<const ::sycl::half*>(inputs[2]->buffer_ptr()) : nullptr;

            return to_ocl_event(stream, run_fc_int8_woq(sycl_queue, barrier, in, wei, dzp, ds, out, M, N, K, group_size, groups_num, out_shape, dzp_scalar));
        } else {
            OPENVINO_THROW("No instance for given types found: ", in_t, " ", wei_t, " ", out_t, " ", ds_t, " ", dzp_t);
        }
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params) {
        auto source_weights_layout = impl_params.get_input_layout(1);
        auto target_weights_layout = source_weights_layout;
        target_weights_layout.format = format::oiyx;

        return std::make_shared<WeightsReorderParams>(source_weights_layout, target_weights_layout);
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return cldnn::make_unique<fully_connected_sycl_example>(engine, config, get_weights_reorder(impl_params));
    }
};

std::unique_ptr<primitive_impl> ExampleImplementationManagerSYCL::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return sycl::fully_connected_sycl_example::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace sycl
}  // namespace cldnn
