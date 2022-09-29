// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_inst.h"
#include "eltwise_inst.h"
#include "quantize_inst.h"
#include "primitive_onednn_base.h"
#include "impls/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct concatenation_onednn : typed_primitive_onednn_impl<concatenation, void, dnnl::concat::primitive_desc, dnnl::concat> {
    using parent = typed_primitive_onednn_impl<concatenation, void, dnnl::concat::primitive_desc, dnnl::concat>;
    using parent::parent;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(concatenation_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        int input_idx = DNNL_ARG_MULTIPLE_SRC;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            args.insert({ input_idx++, input.get_onednn_memory(_pd.src_desc(static_cast<int>(i))) });
        }

        {
            auto& output = instance.output_memory();
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dst_desc())});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    static std::shared_ptr<dnnl::concat::primitive_desc> get_concatenation_descriptor(const kernel_impl_params& impl_params) {
        auto prim = impl_params.typed_desc<concatenation>();

        auto& engine = impl_params.prog.get_engine();
        std::vector<dnnl::memory::desc> input_mds;
        for (size_t i = 0; i < impl_params.input_layouts.size(); i++) {
            input_mds.push_back(onednn::layout_to_memory_desc(impl_params.get_input_layout(i)));
        }
        auto output_md = onednn::layout_to_memory_desc(impl_params.output_layout);
        return std::make_shared<dnnl::concat::primitive_desc>(
            output_md,
            prim->axis,
            input_mds,
            engine.get_onednn_engine());
    }

public:
    static primitive_impl* create(const concatenation_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog.get_engine();
        if (arg.can_be_optimized())
            return new concatenation_onednn(engine);
        auto desc = get_concatenation_descriptor(impl_params);
        auto attr = arg.get_onednn_primitive_attributes();

        std::shared_ptr<void> dummy = nullptr;

        return new concatenation_onednn(engine, dummy, attr, *desc);
    }
};

namespace detail {

attach_concatenation_onednn::attach_concatenation_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
    };
    implementation_map<concatenation>::add(impl_types::onednn, concatenation_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
