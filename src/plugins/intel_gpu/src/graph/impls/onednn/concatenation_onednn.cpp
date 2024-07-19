// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_onednn.hpp"
#include "concatenation_inst.h"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_manager.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <memory>
namespace cldnn {
namespace onednn {

struct concatenation_onednn : typed_primitive_onednn_impl<concatenation, dnnl::concat::primitive_desc, dnnl::concat> {
    using parent = typed_primitive_onednn_impl<concatenation, dnnl::concat::primitive_desc, dnnl::concat>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::concatenation_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<concatenation_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(concatenation_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        int input_idx = DNNL_ARG_MULTIPLE_SRC;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i), _pd.dnnl::primitive_desc_base::src_desc(static_cast<uint8_t>(i)));
            args.insert({input_idx++, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<uint8_t>(i)), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    static std::shared_ptr<dnnl::concat::primitive_desc> get_concatenation_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                      cldnn::engine& engine,
                                                                                      const dnnl::primitive_attr& attr,
                                                                                      const int64_t axis) {
        std::vector<dnnl::memory::desc> input_mds;
        for (size_t i = 0; i < impl_params.input_layouts.size(); i++) {
            input_mds.push_back(onednn::layout_to_memory_desc(impl_params.get_input_layout(i)));
        }
        auto output_md = onednn::layout_to_memory_desc(impl_params.get_output_layout());
        return std::make_shared<dnnl::concat::primitive_desc>(
            engine.get_onednn_engine(),
            output_md,
            axis,
            input_mds,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        if (_prim.get(true) == nullptr) {
            ob << false;
            primitive_impl::save(ob);
            return;
        } else {
            ob << true;
        }

        parent::save(ob);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<concatenation>();
        ob << prim->axis;

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        bool has_prim;
        ib >> has_prim;

        if (!has_prim) {
            primitive_impl::load(ib);
            return;
        }

        parent::load(ib);

        int64_t prim_axis;
        ib >> prim_axis;

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        auto prim_desc = get_concatenation_primitive_descriptor(*impl_params, ib.get_engine(), *_attrs, prim_axis);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::concat(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const concatenation_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        if (impl_params.can_be_optimized())
            return make_unique<concatenation_onednn>(engine, config);
        auto prim = impl_params.typed_desc<concatenation>();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_concatenation_primitive_descriptor(impl_params, impl_params.prog->get_engine(), *attr, prim->axis);

        return cldnn::make_unique<concatenation_onednn>(engine, config, attr, *prim_desc);
    }
};

std::unique_ptr<primitive_impl> ConcatenationImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<concatenation>());
    return onednn::concatenation_onednn::create(static_cast<const concatenation_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::concatenation_onednn)
