// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct pooling_onednn : typed_primitive_onednn_impl<pooling, dnnl::pooling_forward::desc> {
    using parent = typed_primitive_onednn_impl<pooling, dnnl::pooling_forward::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pooling_onednn>(*this);
    }

    static std::shared_ptr<dnnl::pooling_forward::desc> get_pooling_descriptor(const kernel_impl_params& impl_params) {
        auto prim = impl_params.typed_desc<pooling>();

        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.output_layout;

        dnnl::memory::dims stride(prim->stride.begin(), prim->stride.end());
        dnnl::memory::dims kernel(prim->size.begin(), prim->size.end());
        dnnl::memory::dims pad_l(prim->pad.begin(), prim->pad.end());
        dnnl::memory::dims pad_r(prim->pad_end.begin(), prim->pad_end.end());

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        if (prim->global_pooling) {
            for (size_t i = 0; i < kernel.size(); i++)
                kernel[i] = input_md.dims()[2 + i];
        }

        for (size_t i = 0; i < kernel.size(); i++) {
            pad_r[i] = (output_md.dims()[2 + i] - 1) * stride[i] - input_md.dims()[2 + i] + kernel[i] - pad_l[i];
        }

        dnnl::algorithm alg;
        switch (prim->mode) {
            case pooling_mode::average: alg = dnnl::algorithm::pooling_avg; break;
            case pooling_mode::max: alg = dnnl::algorithm::pooling_max; break;
            case pooling_mode::average_no_padding: alg = dnnl::algorithm::pooling_avg_exclude_padding; break;
            default: throw std::runtime_error("unsupported pool mode");
        }

        return std::make_shared<dnnl::pooling_forward::desc>(
            dnnl::prop_kind::forward_inference,
            alg,
            input_md,
            output_md,
            stride,
            kernel,
            pad_l,
            pad_r);
    }

public:
    static primitive_impl* create(const pooling_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog.get_engine();
        auto desc = get_pooling_descriptor(impl_params);
        auto attr = arg.get_onednn_primitive_attributes();
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new pooling_onednn(engine, desc, attr, prim_desc);
    }
};

namespace detail {

attach_pooling_onednn::attach_pooling_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
    };

    implementation_map<pooling>::add(impl_types::onednn, pooling_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
