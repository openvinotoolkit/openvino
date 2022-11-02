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

    DECLARE_OBJECT_TYPE_SERIALIZATION

protected:
    const concatenation_node* _outer;

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

        auto& engine = impl_params.prog->get_engine();
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
    void save(BinaryOutputBuffer& ob) const override {
        if (_prim.get(true) == nullptr) {
            ob << false;
            return;
        } else {
            ob << true;
        }

        parent::save(ob);

        ob << _outer->get_dependencies().size();
        for (auto& input : _outer->get_dependencies()) {
            ob << input->get_output_layout();
        }
        ob << _outer->get_primitive()->axis;
        ob << _outer->get_output_layout();

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
    }

    void load(BinaryInputBuffer& ib) override {
        bool has_prim;
        ib >> has_prim;

        if (!has_prim)
            return;

        parent::load(ib);

        size_t num_deps;
        ib >> num_deps;

        std::vector<dnnl::memory::desc> input_mds;
        for (size_t idx = 0; idx < num_deps; ++idx) {
            layout input_layout = layout(cldnn::data_types::bin, cldnn::format::any, cldnn::tensor());
            ib >> input_layout;
            input_mds.push_back(onednn::layout_to_memory_desc(input_layout));
        }

        int64_t prim_axis;
        ib >> prim_axis;

        layout output_layout = layout(cldnn::data_types::bin, cldnn::format::any, cldnn::tensor());
        ib >> output_layout;
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        auto desc = std::make_shared<dnnl::concat::primitive_desc>(
            output_md,
            prim_axis,
            input_mds,
            ib.get_engine().get_onednn_engine());

        _pd = *desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _prim = dnnl::concat(_pd, prim_cache);
    }

    static primitive_impl* create(const concatenation_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        if (arg.can_be_optimized())
            return new concatenation_onednn(engine);
        auto desc = get_concatenation_descriptor(impl_params);
        auto attr = arg.get_onednn_primitive_attributes();

        std::shared_ptr<void> dummy = nullptr;

        auto new_impl = new concatenation_onednn(engine, dummy, attr, *desc);
        new_impl->_outer = &arg;
        return new_impl;
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
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::concatenation_onednn, cldnn::object_type::CONCATENATION_ONEDNN)
