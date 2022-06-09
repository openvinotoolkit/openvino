// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"
#include "kernel_base.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {
struct reduction_onednn : typed_primitive_onednn_impl<reduce, dnnl::reduction::desc> {
    using parent = typed_primitive_onednn_impl<reduce, dnnl::reduction::desc>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduction_onednn>(*this);
    }

    static std::shared_ptr<dnnl::reduction::desc> get_reduction_descriptor(const reduce_node& arg) {
        auto prim = arg.get_primitive();
        auto& input = arg.get_dependency(0);
        auto input_md = onednn::layout_to_memory_desc(input.get_output_layout());
        auto output_md = onednn::layout_to_memory_desc(arg.get_output_layout());

        float p = 0.f;
        float eps = 0.f;
        dnnl::algorithm alg;
        switch (prim->mode) {
            case reduce_mode::mean: alg = dnnl::algorithm::reduction_mean; break;
            case reduce_mode::max: alg = dnnl::algorithm::reduction_max; break;
            case reduce_mode::min: alg = dnnl::algorithm::reduction_min; break;
            case reduce_mode::sum: alg = dnnl::algorithm::reduction_sum; break;
            case reduce_mode::prod: alg = dnnl::algorithm::reduction_mul; break;
            case reduce_mode::sum_square:
                alg = dnnl::algorithm::reduction_norm_lp_power_p_sum;
                p = 2.0f;
                break;
            case reduce_mode::l1:
                alg = dnnl::algorithm::reduction_norm_lp_sum;
                p = 1.0f;
                break;
            case reduce_mode::l2:
                alg = dnnl::algorithm::reduction_norm_lp_sum;
                p = 2.0f;
                break;
            default: throw std::runtime_error("unsupported reduce mode");
        }

        return std::make_shared<dnnl::reduction::desc>(
            alg,
            input_md,
            output_md,
            p,
            eps);
    }

public:
    static primitive_impl* create(const reduce_node& arg) {
        auto& engine = arg.get_program().get_engine();
        auto desc = get_reduction_descriptor(arg);
        auto attr = arg.get_onednn_primitive_attributes();
        dnnl::primitive_desc prim_desc{&desc->data, attr.get(), engine.get_onednn_engine(), nullptr};

        return new reduction_onednn(arg, desc, attr, prim_desc);
    }
};

namespace detail {

attach_reduction_onednn::attach_reduction_onednn() {
    implementation_map<reduce>::add(impl_types::onednn, reduction_onednn::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv16_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
    });
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
