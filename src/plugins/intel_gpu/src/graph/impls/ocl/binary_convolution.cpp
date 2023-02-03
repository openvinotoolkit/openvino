// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/quantize.hpp"
#include "binary_convolution_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "kernel_runner.h"
#include "kernel_selector/kernels/binary_convolution/binary_convolution_kernel_selector.h"
#include "kernel_selector/kernels/binary_convolution/binary_convolution_params.h"
#include <algorithm>
#include <memory>

namespace cldnn {
namespace ocl {

struct binary_convolution_impl : typed_primitive_impl_ocl<binary_convolution> {
    using parent = typed_primitive_impl_ocl<binary_convolution>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::binary_convolution_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::binary_convolution_params, kernel_selector::binary_convolution_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<binary_convolution_impl>(*this);
    }

    binary_convolution_impl() : parent() {}

    explicit binary_convolution_impl(const binary_convolution_impl& other) : parent(other),
        _split(other._split) {}

    binary_convolution_impl(const binary_convolution_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd) {
        set_node_params(arg);
    }

    void set_node_params(const program_node& arg) override {
        IE_ASSERT(arg.is_type<binary_convolution>());
        const auto& node = arg.as<binary_convolution>();
        _split = node.get_split();
    }

protected:
    bool validate_impl(const typed_primitive_inst<binary_convolution>& instance) const override {
        bool res = true;

        auto data_type = instance.node->input().get_output_layout().data_type;

        // Check whether all memory elements use the same unit type (FP16 or FP32).
        CLDNN_ERROR_DATA_TYPES_MISMATCH(_node_id,
                                        "Input memory",
                                        data_type,
                                        "output memory",
                                        instance.node->get_output_layout().data_type,
                                        "");
        CLDNN_ERROR_DATA_TYPES_MISMATCH_IGNORE_SIGN(_node_id,
                                                    "Input memory",
                                                    data_type,
                                                    "filter memory",
                                                    instance.weights_memory(0)->get_layout().data_type,
                                                    "");

        return res;
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<binary_convolution>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);

        args.weights = instance.weights_memory(split);
        return args;
    }

    int32_t get_split() const override { return _split; }

public:
    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _split;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _split;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<binary_convolution>();
        const auto& weights_layout = (*impl_param.weights_layout).convert_to_weights_layout(false);
        const auto& weights_size = weights_layout.get_tensor();

        const auto& split = primitive->split();
        const auto& groups = primitive->groups;
        const auto& stride = primitive->stride;
        const auto& dilation = primitive->dilation;
        const auto& pad = primitive->pad;

        auto params = get_weights_bias_default_params<kernel_selector::binary_convolution_params>(impl_param, split);
        auto optional_params = get_default_weights_bias_optional_params<kernel_selector::binary_convolution_optional_params>(impl_param.get_program());

        params.pad_value = primitive->pad_value;
        params.out_dt = to_data_type(*primitive->output_data_types[0]);
        params.depthwise_separable_opt = false;
        params.split = static_cast<uint32_t>(split);
        params.groups = static_cast<uint32_t>(groups);
        params.filterSize = {
            (uint32_t)weights_size.spatial[0],
            (uint32_t)weights_size.spatial[1],
            (uint32_t)weights_size.spatial[2],
        };

        uint32_t pad_z = std::max<std::ptrdiff_t>(pad.size() >= 3 ? pad[pad.size() - 3] : 0, 0);
        uint32_t pad_y = std::max<std::ptrdiff_t>(pad.size() >= 2 ? pad[pad.size() - 2] : 0, 0);
        uint32_t pad_x = std::max<std::ptrdiff_t>(pad.size() >= 1 ? pad[pad.size() - 1] : 0, 0);
        params.padding = {pad_x, pad_y, pad_z};

        uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
        uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
        uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;
        params.stride = {stride_x, stride_y, stride_z};

        uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;
        params.dilation = {dilation_x, dilation_y, dilation_z};

        const auto& tuning_config = impl_param.get_program().get_options().get<build_option_type::tuning_config>();

        if (tuning_config->config.mode == tuning_mode::tuning_tune_and_cache ||
            tuning_config->config.mode == tuning_mode::tuning_retune_and_cache) {
            optional_params.tuningParams.runner =
                std::make_shared<gpu::kernel_runner>(impl_param.get_program().get_engine(), impl_param.get_program().get_id(), true);
        }

        return {params, optional_params};
    }

private:
    int32_t _split;
};

namespace detail {

attach_binary_convolution_impl::attach_binary_convolution_impl() {
    implementation_map<binary_convolution>::add(impl_types::ocl, typed_primitive_impl_ocl<binary_convolution>::create<binary_convolution_impl>, {
        std::make_tuple(data_types::bin, format::b_fs_yx_32fp),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::binary_convolution_impl)
