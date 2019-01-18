/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef RNN_PD_HPP
#define RNN_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "memory_pd.hpp"
#include "primitive_desc.hpp"

namespace mkldnn {
namespace impl {

// struct rnn_fwd_pd_t;

struct rnn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_pd_t(mkldnn::impl::engine_t *engine, const rnn_desc_t *adesc,
            const primitive_attr_t *attr, const rnn_pd_t *hint_pd)
        : primitive_desc_t(engine, attr, primitive_kind::rnn)
        , desc_(*adesc)
        , hint_pd_(hint_pd) {}
    virtual ~rnn_pd_t() {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }
    virtual void init_info() override { init_info_rnn(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::rnn_d: *(const rnn_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    inline bool is_training() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::backward);
    }

    inline bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    inline size_t ws_states_size() {
        return (size_t)(L() + 1) * D() * (T() + 1) * S() * MB() * S_GLD();
    }

    inline size_t ws_diff_states_size() {
        return (size_t)(L() + 1) * D() * (T() + 1) * (S() + 1) * MB() * S_GLD();
    }

    inline size_t ws_weights_layer_size() {
        size_t ld = is_fwd() ? G_GLD() : S_GLD();
        size_t not_ld =  is_fwd() ? SLC() : G() * DIC();
        return (size_t)(L() * D() * ld * not_ld);
    }

    inline size_t ws_weights_iter_size() {
        size_t ld = is_fwd() ? G_GLD() : S_GLD();
        size_t not_ld =  is_fwd() ? SIC() : G() * DIC();
        return (size_t)(L() * D() * ld * not_ld);
    }

    inline size_t ws_diff_weights_layer_size() {
        return (size_t)(L() * D() * SLC() * GC());
    }

    inline size_t ws_diff_weights_iter_size() {
        return (size_t)(L() * D() * SIC() * GC());
    }

    inline size_t ws_gates_size() {
        return (size_t) L() * D() * T() * MB() * GC();
    }

    inline size_t ws_cell_comp_size() {
        return (size_t)is_lbr() * MB() * GC();
    }

    inline size_t ws_grid_comp_size() {
        return (size_t)is_lbr() * is_training() * L() * D() * T() * MB() * DIC();
    }

    inline int ws_per_cell() {
        return is_lbr() *  MB() * DIC();
    }

    // returns the scratchpad size if use_workspace is true
    // returns the workspace size if use_workspace is false,
    // and all scratchpad boolean are false
    inline size_t set_offsets( bool use_workspace,
        size_t &ws_gates_offset, size_t &ws_states_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        bool use_ws_cell_comp, size_t &ws_cell_comp_offset,
        bool copy_weights_layer_, size_t &ws_weights_layer_offset,
        bool copy_weights_iter_, size_t &ws_weights_iter_offset,
        bool copy_diff_weights_layer, size_t &ws_diff_weights_layer_offset,
        bool copy_diff_weights_iter, size_t &ws_diff_weights_iter_offset) {
        const size_t page_size = 4096; // 2097152;
        size_t current_offset;

        /* Mandatory workspaces: go to workspace if use_workspace, scratchpad otherwise */
        current_offset = 0;  // assumes the workspace base pointer is page aligned
        ws_gates_offset = current_offset;
        current_offset += ws_gates_size();

        current_offset = utils::rnd_up(current_offset, page_size);
        ws_states_offset = current_offset;
        current_offset += ws_states_size();

        current_offset = utils::rnd_up(current_offset, page_size);
        ws_diff_states_offset = current_offset;
        current_offset += ws_diff_states_size();

        current_offset = utils::rnd_up(current_offset, page_size);
        ws_grid_comp_offset = current_offset;
        current_offset += ws_grid_comp_size();

        // ws_cell_comp is optional
        if (use_ws_cell_comp) {
            current_offset = utils::rnd_up(current_offset, page_size);
            ws_cell_comp_offset = current_offset;
            current_offset += ws_cell_comp_size();
        }

        /* Optional scratchpads */
        // Assumes the scratchpad base pointer is page aligned.
        // If use_workspace, the following goes to scratchpad alone,
        // otherwise, all goes to scratchpad and continue incrementing offset
        current_offset = use_workspace ? 0 : current_offset;

        if (copy_weights_layer_) {
            current_offset = utils::rnd_up(current_offset, page_size);
            ws_weights_layer_offset = current_offset;
            current_offset += ws_weights_layer_size();
        }

        if (copy_weights_iter_) {
            current_offset = utils::rnd_up(current_offset, page_size);
            ws_weights_iter_offset = current_offset;
            current_offset += ws_weights_iter_size();
        }

        if (copy_diff_weights_layer) {
            current_offset = utils::rnd_up(current_offset, page_size);
            ws_diff_weights_layer_offset = current_offset;
            current_offset += ws_diff_weights_layer_size();
        }

        if (copy_diff_weights_iter) {
            current_offset = utils::rnd_up(current_offset, page_size);
            ws_diff_weights_iter_offset = current_offset;
            current_offset += ws_diff_weights_iter_size();
        }

        return current_offset;
    }

    inline size_t get_ws_size() {
        size_t ws_gates_offset, ws_states_offset,
            ws_diff_states_offset,ws_grid_comp_offset,
            ws_cell_comp_offset, ws_weights_layer_offset,
            ws_weights_iter_offset, ws_diff_weights_layer_offset,
            ws_diff_weights_iter_offset;
        return set_offsets( false,
                     ws_gates_offset, ws_states_offset,
                     ws_diff_states_offset, ws_grid_comp_offset,
                     is_lbr(), ws_cell_comp_offset,
                     false, ws_weights_layer_offset,
                     false, ws_weights_iter_offset,
                     false, ws_diff_weights_layer_offset,
                     false, ws_diff_weights_iter_offset);
    }

    inline size_t get_scratchpad_size(bool use_workspace) {
        size_t ws_gates_offset, ws_states_offset,
            ws_diff_states_offset,ws_grid_comp_offset,
            ws_cell_comp_offset, ws_weights_layer_offset,
            ws_weights_iter_offset, ws_diff_weights_layer_offset,
            ws_diff_weights_iter_offset;
        return set_offsets(use_workspace,
                     ws_gates_offset, ws_states_offset,
                     ws_diff_states_offset, ws_grid_comp_offset,
                     false, ws_cell_comp_offset,
                     false, ws_weights_layer_offset,
                     false, ws_weights_iter_offset,
                     false, ws_diff_weights_layer_offset,
                     false, ws_diff_weights_iter_offset);
    }

    int T() const { return desc_.src_layer_desc.dims[0]; }
    int MB() const { return desc_.src_layer_desc.dims[1]; }

    int L() const { return desc_.weights_layer_desc.dims[0]; }
    int D() const { return desc_.weights_layer_desc.dims[1]; }

    int SIC() const { return desc_.weights_iter_desc.dims[2]; }

    int SLC() const { return desc_.weights_layer_desc.dims[2]; }
    int G() const { return desc_.weights_layer_desc.dims[3]; }
    int DIC() const { return desc_.weights_layer_desc.dims[4]; }

    int DLC() const { return desc_.dst_layer_desc.dims[2]; }

    int get_good_ld(int dim){
        // we want matrices leading dimentions to be 64-byte aligned,
        // and not divisible by 256 to avoid 4K aliasing effects
        int ld = utils::rnd_up(dim, (int)(64/sizeof(float)));
        return (ld % 256 == 0) ? ld + 64/sizeof(float) : ld;
    }

    int WIC() {
        // wic will be the leading dimension of our B matrices
        return get_good_ld(nstl::max(SLC(), nstl::max(SIC(), DIC())));
    }

    int GC() {
        // gc will be the leading dimension of our C matrices
        return get_good_ld(G() * DIC());
    }

    /* replacement functions for meaningless WIC and GC:
       - LD stands for leading dimension
       - GLD stands for good leading dimension
       - NLD stands for not leading dimension (so the other dim)
    */
    int G_GLD() {
        // good leading dimension for the gates
        // C matrices for fwd, B matrices for bwd
        return get_good_ld(G() * DIC());
    }

    int S_GLD() {
        // good leading dimension for the states
        // B matrices for fwd, B matrices for bwd_w, C matrices for bwd_d
        return get_good_ld(nstl::max(SLC(), nstl::max(SIC(), DIC())));
    }

    int W_GLD() {
        // good leading dimension for the weights
        return is_fwd() ? G_GLD() : S_GLD();
    }

    int DW_GLD() {
        // good leading dimension for the diff weights
        return weights_copy_enabled() ? G_GLD() : G() * DIC();
    }

    int weights_copy_enabled() { return (T() > 1); }

    int get_weights_ld(int feature_dim) {
        return is_fwd() ? G() * DIC() : feature_dim;
    }

    int get_weights_nld(int feature_dim) {
        return !(is_fwd()) ? G() * DIC() : feature_dim;
    }

    int WL_LD() {
        return get_weights_ld(SLC());
    }

    int WL_GLD() {
        return weights_copy_enabled() ? get_good_ld(WL_LD()) : WL_LD();
    }

    int WI_LD() {
        return get_weights_ld(SIC());
    }

    int WI_GLD() {
        return weights_copy_enabled() ? get_good_ld(WI_LD()) : WI_LD();
    }

    int DWL_LD() {
        return G() * DIC();
    }

    int DWL_GLD() {
        return weights_copy_enabled() ? get_good_ld(DWL_LD()) : DWL_LD();
    }

    int DWI_LD() {
        return G() * DIC();
    }

    int DWI_GLD() {
        return weights_copy_enabled() ? get_good_ld(DWI_LD()) : DWI_LD();
    }

    int WL_NLD() {
        return get_weights_nld(SLC());
    }

    int WI_NLD() {
        return get_weights_nld(SIC());
    }

    int DWL_NLD() {
        return SLC();
    }

    int DWI_NLD() {
        return SIC();
    }

    int S() const { return mkldnn_rnn_cell_get_states_count(&desc_.cell_desc); }

    bool with_bias() const {
        return !memory_desc_wrapper(desc_.bias_desc).is_zero();
    }

    bool with_src_iter() const {
        return !(memory_desc_wrapper(desc_.src_iter_desc).is_zero());
    }

    bool with_dst_iter() const {
        return !memory_desc_wrapper(desc_.dst_iter_desc).is_zero();
    }

    mkldnn::impl::alg_kind_t cell_kind() const {
        return desc_.cell_desc.cell_kind;
    }
    mkldnn::impl::alg_kind_t activation_kind() const {
        return desc_.cell_desc.activation_kind;
    }

    bool is_lbr() const {
        return cell_kind() == mkldnn_gru_linear_before_reset;
    }

    mkldnn_rnn_direction_t direction() const { return desc_.direction; }

protected:
    rnn_desc_t desc_;
    const rnn_pd_t *hint_pd_;
};

struct rnn_fwd_pd_t : public rnn_pd_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    using rnn_pd_t::rnn_pd_t;
    virtual ~rnn_fwd_pd_t() {}

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (index == 0) return src_pd(0);
        if (with_src_iter() && index == 1) return src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 3) return weights_pd(index);

        return nullptr;
    }

    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return dst_pd(0);
        if (with_dst_iter() && index == 1) return dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (is_training() && index == 0) return workspace_pd();

        return nullptr;
    }

    virtual int n_inputs() const override {
        return 3 + with_bias() + with_src_iter();
    }

    virtual int n_outputs() const override {
        return 1 + with_dst_iter() + is_training();
    }

    int ws_idx() const { return 1 + with_dst_iter(); }
};

struct rnn_bwd_pd_t : public rnn_pd_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_bwd_pd_t hint_class;

    using rnn_pd_t::rnn_pd_t;
    virtual ~rnn_bwd_pd_t() {}

    virtual const memory_pd_t *input_pd(int index = 0) const override {
        if (index == 0) return src_pd(0);
        if (with_src_iter() && index == 1) return src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 2) return weights_pd(index);
        if (with_bias() && index == 2) return weights_pd(2);
        index = index - 2 - with_bias();

        if (index == 0) return dst_pd(0);
        if (with_dst_iter() && index == 1) return dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (index == 0) return diff_dst_pd(0);
        if (with_dst_iter() && index == 1) return diff_dst_pd(1);
        index = index - 1 - with_dst_iter();

        if (index == 0) return workspace_pd();

        return nullptr;
    }

    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return diff_src_pd(0);
        if (with_src_iter() && index == 1) return diff_src_pd(1);
        index = index - 1 - with_src_iter();

        if (index < 3) return diff_weights_pd(index);

        return nullptr;
    }

    virtual int n_inputs() const override {
        return 6 + with_src_iter() + with_bias() + 2 * with_dst_iter();
    }
    virtual int n_outputs() const override {
        return 3 + with_src_iter() + with_bias();
    }

    int ws_idx() const {
        return 5 + with_src_iter() + with_bias() + 2 * with_dst_iter();
    }
};
}
}

#endif
