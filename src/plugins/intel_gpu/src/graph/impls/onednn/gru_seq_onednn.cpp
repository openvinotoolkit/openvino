// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/onednn/utils.hpp"
#include "gru_seq_inst.h"
#include "primitive_onednn_base.h"
#include "gru_seq_onednn.hpp"
#include "registry/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn::onednn {

struct gru_seq_onednn : typed_primitive_onednn_impl<gru_seq> {
    using parent = typed_primitive_onednn_impl<gru_seq>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::gru_seq_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<gru_seq_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(gru_seq_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;

        {
            int i = 0;
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)));
            auto mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)), offset);
            args.insert({DNNL_ARG_SRC_LAYER, mem});
        }

        {
            int i = 1;
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)));
            auto mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(static_cast<int>(i)), offset);
            args.insert({DNNL_ARG_SRC_ITER, mem});
        }

        {
            int i = 2;
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::weights_desc(0));
            auto mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::weights_desc(0), offset);
            args.insert({DNNL_ARG_WEIGHTS_LAYER, mem});
        }

        {
            int i = 3;
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::weights_desc(1));
            auto mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::weights_desc(1), offset);
            args.insert({DNNL_ARG_WEIGHTS_ITER, mem});
        }

        {//bias
            int i = 4;
            auto& input = instance.input_memory(i);
            auto offset = onednn::get_offset(instance.get_input_layout(i),
                                             _pd.dnnl::primitive_desc_base::weights_desc(2));
            auto mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::weights_desc(2), offset);
            args.insert({DNNL_ARG_BIAS, mem});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            auto mem = output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset);
            args.insert({DNNL_ARG_DST_LAYER, mem});
        }

        {
            auto& output = instance.output_memory(1);
            auto offset = onednn::get_offset(instance.get_output_layout(1), _pd.dnnl::primitive_desc_base::dst_desc(1));
            auto mem = output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(1), offset);
            args.insert({DNNL_ARG_DST_ITER, mem});
        }

        return args;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        const auto weights_layout_idx = 2;
        auto source_weights_layout = impl_params.get_input_layout(weights_layout_idx);
        auto target_weights_layout = impl_params.get_input_layout(weights_layout_idx);
        auto W_desc = onednn::layout_to_memory_desc(source_weights_layout);
        auto grouped_weights = format::is_grouped(source_weights_layout.format);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            W_desc,
                                                            W_desc,
                                                            false,
                                                            grouped_weights);
    }

    static std::shared_ptr<dnnl::lbr_gru_forward::primitive_desc> get_gru_primitive_descriptor(const kernel_impl_params& impl_params, cldnn::engine& engine,
                                                                                               const dnnl::primitive_attr& attr,
                                                                                               ov::op::RecurrentSequenceDirection direction) {
        auto prim = impl_params.typed_desc<gru_seq>();
        auto num_dir = static_cast<size_t>(prim->num_directions());
        assert(prim->linear_before_reset);
        const auto& src_shape = impl_params.get_input_layout(0).get_shape();
        auto mod_src_shape = src_shape;
        std::swap(mod_src_shape[0], mod_src_shape[1]);
        auto input_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(0).clone_with_other_shape(mod_src_shape), dnnl::memory::format_tag::abc);
        auto initial_hidden_shape_mod = impl_params.get_input_layout(1).get_shape();
        initial_hidden_shape_mod = { 1, num_dir, initial_hidden_shape_mod[0], initial_hidden_shape_mod[2] };
        auto initial_hidden =  onednn::layout_to_memory_desc(impl_params.get_input_layout(1).clone_with_other_shape(initial_hidden_shape_mod));
        auto W_shape_mod = impl_params.get_input_layout(2).get_shape();
        W_shape_mod = {1, num_dir, W_shape_mod[2], 3, W_shape_mod[1]/3};
        auto w_layout = impl_params.get_input_layout(2).clone_with_other_shape(W_shape_mod);
        w_layout.format = cldnn::format::bfzyx;
        auto W_md = onednn::layout_to_memory_desc(w_layout);
        auto R_shape_mod = impl_params.get_input_layout(3).get_shape();
        R_shape_mod = {1, num_dir, R_shape_mod[2], 3, R_shape_mod[1]/3};
        auto r_layout = impl_params.get_input_layout(3).clone_with_other_shape(R_shape_mod);
        r_layout.format = cldnn::format::bfzyx;
        auto R_md = onednn::layout_to_memory_desc(r_layout);
        auto B_shape_mod = impl_params.get_input_layout(4).get_shape();
        B_shape_mod = {1, num_dir, 4, B_shape_mod[1]/4};
        auto b_layout = impl_params.get_input_layout(4).clone_with_other_shape(B_shape_mod);
        b_layout.format = cldnn::format::bfyx;
        auto B_md = onednn::layout_to_memory_desc(b_layout);
        auto out_shape = impl_params.get_output_layout().get_shape();
        out_shape = {out_shape[2], out_shape[0], num_dir*out_shape[3], 1};
        auto output_md = onednn::layout_to_memory_desc(impl_params.get_output_layout().clone_with_other_shape(out_shape), dnnl::memory::format_tag::abc);
        auto output1_md = onednn::layout_to_memory_desc(impl_params.get_output_layout().clone_with_other_shape(initial_hidden_shape_mod));
        OPENVINO_ASSERT(input_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the input memory descriptor of onednn gru_seq cannot be 'any'.");
        OPENVINO_ASSERT(output_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the output memory descriptor of onednn gru_seq cannot be 'any'.");

        dnnl::memory::desc emptyMemDescriptor;
        dnnl::rnn_direction gru_desc_dir;
        if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
            gru_desc_dir = dnnl::rnn_direction::unidirectional_left2right;
        } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
            gru_desc_dir = dnnl::rnn_direction::unidirectional_right2left;
        } else {
            gru_desc_dir = dnnl::rnn_direction::bidirectional_concat;
        }
        auto eng = engine.get_onednn_engine();
        return std::make_shared<dnnl::lbr_gru_forward::primitive_desc>(
            eng,
            dnnl::prop_kind::forward_inference,
            gru_desc_dir,
            input_md,
            initial_hidden,
            W_md,
            R_md,
            B_md,
            output_md,
            output1_md);
    }

public:
void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
    parent::save(ob);
    const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
    auto prim = impl_params->typed_desc<gru_seq>();
    ob << prim->linear_before_reset;
    ob << static_cast<int>(prim->direction);
    std::vector<uint8_t> prim_cache;
    prim_cache = _prim.get_cache_blob();
    ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        ov::op::RecurrentSequenceDirection direction;
        int dir;
        bool linear_before_reset;
        ib >> linear_before_reset;
        OPENVINO_ASSERT(linear_before_reset);
        ib >> dir;
        direction = static_cast<ov::op::RecurrentSequenceDirection>(dir);
        dnnl::rnn_direction gru_desc_dir;
        unsigned long num_dir = 1;
        if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
            gru_desc_dir = dnnl::rnn_direction::unidirectional_left2right;
        } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
            gru_desc_dir = dnnl::rnn_direction::unidirectional_right2left;
        } else {
            gru_desc_dir = dnnl::rnn_direction::bidirectional_concat;
            num_dir = 2;
        }
        auto mod_src_shape = impl_params->get_input_layout(0).get_shape();
        std::swap(mod_src_shape[0], mod_src_shape[1]);
        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0).clone_with_other_shape(mod_src_shape), dnnl::memory::format_tag::abc);
        auto initial_hidden_shape_mod = impl_params->get_input_layout(1).get_shape();
        initial_hidden_shape_mod = {1, num_dir, initial_hidden_shape_mod[0], initial_hidden_shape_mod[2]};
        auto initial_hidden_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(1).clone_with_other_shape(initial_hidden_shape_mod));
        auto W_shape_mod = impl_params->get_input_layout(2).get_shape();
        W_shape_mod = {1, num_dir, W_shape_mod[1], 3, W_shape_mod[2]/3};
        auto w_layout = impl_params->get_input_layout(2).clone_with_other_shape(W_shape_mod);
        w_layout.format = cldnn::format::bfzyx;
        auto W_md = onednn::layout_to_memory_desc(w_layout);
        auto R_shape_mod = impl_params->get_input_layout(3).get_shape();
        R_shape_mod = {1, num_dir, R_shape_mod[1], 3, R_shape_mod[2]/3};
        auto r_layout = impl_params->get_input_layout(3).clone_with_other_shape(R_shape_mod);
        r_layout.format = cldnn::format::bfzyx;
        auto R_md = onednn::layout_to_memory_desc(r_layout);
        auto B_shape_mod = impl_params->get_input_layout(4).get_shape();
        B_shape_mod = {1, num_dir, 4, B_shape_mod[1]/4};
        auto B_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(4).clone_with_other_shape(B_shape_mod));
        auto out_shape = impl_params->get_output_layout().get_shape();
        out_shape = {out_shape[2], out_shape[0], out_shape[3]*num_dir};
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout().clone_with_other_shape(out_shape), dnnl::memory::format_tag::abc);
        auto output1_md = onednn::layout_to_memory_desc(impl_params->get_output_layout(1).clone_with_other_shape(initial_hidden_shape_mod));

        auto prim_desc = std::make_shared<dnnl::lbr_gru_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            gru_desc_dir,
            input_md,
            initial_hidden_md,
            W_md,
            R_md,
            B_md,
            output_md,
            output1_md);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;
        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const gru_seq_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto direction = arg.direction();
        auto prim_desc = get_gru_primitive_descriptor(impl_params, engine, *attr, direction);
        return std::make_unique<gru_seq_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
    }
};

std::unique_ptr<primitive_impl> GRUSeqImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    assert(node.is_type<gru_seq>());
    return onednn::gru_seq_onednn::create(static_cast<const gru_seq_node&>(node), params);
}

}  // namespace cldnn::onednn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::gru_seq_onednn)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gru_seq)
