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

    static std::shared_ptr<dnnl::concat::primitive_desc> get_concatenation_descriptor(const concatenation_node& arg) {
        auto prim = arg.get_primitive();

        auto& engine = arg.get_program().get_engine();
        std::vector<dnnl::memory::desc> input_mds;
        for (auto& input : arg.get_dependencies()) {
            input_mds.push_back(onednn::layout_to_memory_desc(input->get_output_layout()));
        }
        auto output_md = onednn::layout_to_memory_desc(arg.get_output_layout());
        return std::make_shared<dnnl::concat::primitive_desc>(
            output_md,
            prim->axis,
            input_mds,
            engine.get_onednn_engine());
    }

public:
    static primitive_impl* create(const concatenation_node& arg) {
        auto desc = get_concatenation_descriptor(arg);
        auto attr = arg.get_onednn_primitive_attributes();

        std::shared_ptr<void> dummy = nullptr;

        return new concatenation_onednn(arg, dummy, attr, *desc);
    }
};

namespace detail {

attach_concatenation_onednn::attach_concatenation_onednn() {
    implementation_map<concatenation>::add(impl_types::onednn, concatenation_onednn::create, {
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

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv4_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv4_fsv4),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv8_fsv4),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv8_fsv4),
    });
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn
