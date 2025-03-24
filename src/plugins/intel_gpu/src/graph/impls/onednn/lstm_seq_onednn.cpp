// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_seq_inst.h"
#include "primitive_onednn_base.h"
#include "lstm_seq_onednn.hpp"
#include "registry/implementation_map.hpp"

#include "kernel_selector_common.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct lstm_seq_onednn : typed_primitive_onednn_impl<lstm_seq> {
    using parent = typed_primitive_onednn_impl<lstm_seq>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::lstm_seq_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<lstm_seq_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(lstm_seq_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;
        std::vector<std::vector<unsigned int>> dnnl_arg{{DNNL_ARG_SRC_LAYER, DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C}, {DNNL_ARG_WEIGHTS_LAYER,
            DNNL_ARG_WEIGHTS_ITER, DNNL_ARG_BIAS}, {DNNL_ARG_DST_LAYER, DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C}};

        for (int i = 0; i < 3; i++) {
            for (int j = 0 ; j < 3; j++) {
                dnnl::memory mem;
                switch (i) {
                    case 0:
                        {
                            auto& input = instance.input_memory(j);
                            auto offset = onednn::get_offset(instance.get_input_layout(j), _pd.dnnl::primitive_desc_base::src_desc(j));
                            mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(j), offset);
                            break;
                        }
                    case 1:
                        {
                            auto& input = instance.input_memory(3+j);
                            auto offset = onednn::get_offset(instance.get_input_layout(3+j), _pd.dnnl::primitive_desc_base::weights_desc(j));
                            mem = input.get_onednn_memory(_pd.dnnl::primitive_desc_base::weights_desc(j), offset);
                            break;
                        }
                    case 2:
                        {
                            auto& output = instance.output_memory(j);
                            auto offset = onednn::get_offset(instance.get_output_layout(j), _pd.dnnl::primitive_desc_base::dst_desc(j));
                            mem = output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(j), offset);
                            break;
                        }
                    default:
                        break;
                }
                args.insert({dnnl_arg[i][j], mem});
            }
        }
        return args;
    }

    static cldnn::layout get_reorder_layout(const kernel_impl_params& impl_params, size_t layout_nr) {
        auto weights_shape = impl_params.get_input_layout(layout_nr).get_shape();
        auto target_weights_layout = impl_params.get_input_layout(layout_nr);
        target_weights_layout.format = cldnn::format::bfzyx;
        auto layout = target_weights_layout.clone_with_other_shape(ov::Shape{weights_shape[0], weights_shape[1], weights_shape[2], 1, 1});
        return layout;
    }

    static std::shared_ptr<WeightsReorderParams> get_weights_reorder(const kernel_impl_params& impl_params, const dnnl::primitive_desc& pd) {
        const auto weights_layout_idx = 3;
        auto source_weights_layout = impl_params.get_input_layout(weights_layout_idx);
        auto target_weights_layout = get_reorder_layout(impl_params, weights_layout_idx);
        auto W_desc = onednn::layout_to_memory_desc(source_weights_layout);
        auto grouped_weights = format::is_grouped(source_weights_layout.format);

        return std::make_shared<WeightsReorderParamsOneDNN>(source_weights_layout,
                                                            target_weights_layout,
                                                            W_desc,
                                                            W_desc,
                                                            false,
                                                            grouped_weights);
    }
    static std::shared_ptr<dnnl::lstm_forward::primitive_desc> get_lstm_primitive_descriptor(const kernel_impl_params& impl_params, cldnn::engine& engine,
                                                                                             const dnnl::primitive_attr& attr,
                                                                                             ov::op::RecurrentSequenceDirection direction) {
        auto prim = impl_params.typed_desc<lstm_seq>();
        auto num_dir = static_cast<size_t>(prim->num_directions());
        const auto& src_shape = impl_params.get_input_layout(0).get_shape();
        auto mod_src_shape = src_shape;
        std::swap(mod_src_shape[0], mod_src_shape[1]);
        auto input_md = onednn::layout_to_memory_desc(impl_params.get_input_layout(0).clone_with_other_shape(mod_src_shape), dnnl::memory::format_tag::abc);
        auto initial_hidden_shape_mod = impl_params.get_input_layout(1).get_shape();
        initial_hidden_shape_mod = { 1, num_dir, initial_hidden_shape_mod[0], initial_hidden_shape_mod[2] };
        auto initial_hidden =  onednn::layout_to_memory_desc(impl_params.get_input_layout(1).clone_with_other_shape(initial_hidden_shape_mod));
        auto initial_cell =  onednn::layout_to_memory_desc(impl_params.get_input_layout(2).clone_with_other_shape(initial_hidden_shape_mod));
        auto W_shape_mod = impl_params.get_input_layout(3).get_shape();
        W_shape_mod = {1, num_dir, W_shape_mod[2], 4, W_shape_mod[1]/4};
        auto w_layout = impl_params.get_input_layout(3).clone_with_other_shape(W_shape_mod);
        w_layout.format = cldnn::format::bfzyx;
        auto W_md = onednn::layout_to_memory_desc(w_layout);
        auto R_shape_mod = impl_params.get_input_layout(4).get_shape();
        R_shape_mod = {1, num_dir, R_shape_mod[2], 4, R_shape_mod[1]/4};
        auto r_layout = impl_params.get_input_layout(4).clone_with_other_shape(R_shape_mod);
        r_layout.format = cldnn::format::bfzyx;
        auto R_md = onednn::layout_to_memory_desc(r_layout);
        auto B_shape_mod = impl_params.get_input_layout(5).get_shape();
        B_shape_mod = {1, num_dir, 4, B_shape_mod[1]/4};
        auto b_layout = impl_params.get_input_layout(5).clone_with_other_shape(B_shape_mod);
        b_layout.format = cldnn::format::bfyx;
        auto B_md = onednn::layout_to_memory_desc(b_layout);
        auto out_shape = impl_params.get_output_layout().get_shape();
        out_shape = {out_shape[2], out_shape[0], out_shape[3]*num_dir};
        auto output_md = onednn::layout_to_memory_desc(impl_params.get_output_layout().clone_with_other_shape(out_shape), dnnl::memory::format_tag::abc);
        auto output1_md = onednn::layout_to_memory_desc(impl_params.get_output_layout(1).clone_with_other_shape(initial_hidden_shape_mod));
        auto output2_md = onednn::layout_to_memory_desc(impl_params.get_output_layout(2).clone_with_other_shape(initial_hidden_shape_mod));
        OPENVINO_ASSERT(input_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the input memory descriptor of onednn lstm_seq cannot be 'any'.");
        OPENVINO_ASSERT(output_md.get_format_kind() != dnnl::memory::format_kind::any,
                        "[GPU] The format kind of the output memory descriptor of onednn lstm_seq cannot be 'any'.");

        auto eng = engine.get_onednn_engine();
        dnnl::rnn_direction lstm_desc_dir;
        if (direction == ov::op::RecurrentSequenceDirection::FORWARD) {
            lstm_desc_dir = dnnl::rnn_direction::unidirectional_left2right;
        } else if (direction == ov::op::RecurrentSequenceDirection::REVERSE) {
            lstm_desc_dir = dnnl::rnn_direction::unidirectional_right2left;
        } else {
            lstm_desc_dir = dnnl::rnn_direction::bidirectional_concat;
        }
        return std::make_shared<dnnl::lstm_forward::primitive_desc>(
            eng,
            dnnl::prop_kind::forward_inference,
            lstm_desc_dir,
            input_md,
            initial_hidden,
            initial_cell,
            W_md,
            R_md,
            B_md,
            output_md,
            output1_md,
            output2_md);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0));
        auto initial_hidden_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(1));
        auto initial_cell_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(2));
        auto W_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(3));
        auto R_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(4));
        auto B_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(5));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());
        auto output2_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());
        auto prim_desc = std::make_shared<dnnl::lstm_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::rnn_direction::undef,
            input_md,
            initial_hidden_md,
            initial_cell_md,
            W_md,
            R_md,
            B_md,
            output_md,
            output_md,
            output2_md);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;
        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const lstm_seq_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto direction = arg.direction();
        auto prim_desc = get_lstm_primitive_descriptor(impl_params, engine, *attr, direction);
        return std::make_unique<lstm_seq_onednn>(engine, config, attr, *prim_desc, get_weights_reorder(impl_params, *prim_desc));
    }
};

std::unique_ptr<primitive_impl> LSTMSeqImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    assert(node.is_type<lstm_seq>());
    return onednn::lstm_seq_onednn::create(static_cast<const lstm_seq_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::lstm_seq_onednn)
