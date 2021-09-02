// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/utils.hpp"

#include "data_inst.h"
#include "reorder_inst.h"
#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "fully_connected_inst.h"
#include "detection_output_inst.h"
#include "binary_convolution_inst.h"
#include "lstm_gemm_inst.h"
#include "generic_layer.hpp"
#include "non_max_suppression_inst.h"
#include "region_yolo_inst.h"

#include "kernel_selector_common.h"
#include "kernel_selector_helper.h"

#include <vector>
#include <memory>
#include <map>
#include <utility>

namespace cldnn {

class primitive_inst;

// this class is used for both static and dynamic reordering of data withing network.
// static reordering is done for cldnn::data (i.e. immutable) primitives via internal network
//  - its done once before network build by running reorder in separate network and fetching its result.
// dynamic reordering is done for cldnn::input_layout (i.e. unknown data during network building)
//  - its done by inserting extra reorder into target topology.
//
// this class does not choose whether there's a need for static or dynamic optimization.
// it's programmers responsiblity to choose between 'get_reorder', which creates reorder to best format
// for given primitive (or nullptr if it's already optimal) and user shall insert it into it's own topology.
//  (note: layout_optimizer has internal caching mechanism, so if there's already reorder added for given (mem,format)
//   pair during 'get_reorder' call, it will be reused).

class reorder_factory {
public:
    // pair.first is reorder (may be nullptr if reorder is not needed), pair.second tells if returned reorder was cached
    // (no need to add it to 'ouputs' etc.) for pair.first == nullptr, pair.second == true
    std::pair<std::shared_ptr<reorder>, bool> get_reorder(primitive_id src_id,
                                                          const layout& in_layout,
                                                          const layout& out_layout);

    std::vector<std::pair<std::shared_ptr<primitive>, bool>> get_weights_reorder(
        primitive_id input_id,
        const layout& old_layout,
        const kernel_selector::weights_reorder_params& reorder_params);

private:
    struct cache_key {
        primitive_id data_source;
        layout expected_layout;

        friend bool operator==(cache_key const& lhs, cache_key const& rhs) {
            return lhs.data_source == rhs.data_source && lhs.expected_layout == rhs.expected_layout;
        }

        friend bool operator!=(cache_key const& lhs, cache_key const& rhs) { return !(lhs == rhs); }

        friend bool operator<(cache_key const& lhs, cache_key const& rhs) {
            if (lhs.data_source != rhs.data_source)
                return (lhs.data_source < rhs.data_source);
            return lhs.expected_layout < rhs.expected_layout;
        }
    };

    std::map<cache_key, std::shared_ptr<reorder>> _cached_reorders;
    std::map<cache_key, std::shared_ptr<generic_layer>> _cached_generic_reorders;
};

class layout_optimizer {
public:
    enum class optimization_attributes_type {
        splitted_convolution,
        group_convolution,
        deformable_convolution,
        bfyx_only_layer,
        fs_b_yx_fsv32_network,
        b_fs_zyx_fsv32_network,
        b_fs_yx_fsv16_network,
        b_fs_zyx_fsv16_network,
        bs_fs_yx_bsv16_fsv16_network
    };

    struct optimization_attributes {
        int32_t splitted_convolution = 0;
        int32_t group_convolution = 0;
        int32_t deformable_convolution = 0;
        int32_t bfyx_only_layer = 0;
        int32_t fs_b_yx_fsv32_network = 0;
        int32_t b_fs_zyx_fsv32_network = 0;
        int32_t b_fs_yx_fsv16_network = 0;
        int32_t b_fs_zyx_fsv16_network = 0;
        int32_t bs_fs_yx_bsv16_fsv16_network = 0;
    };

private:
    optimization_attributes _optimization_attributes;
    // TODO: Remove once we will get full support for input/output padding in all primitive implementations.
    bool _output_size_handling_enabled;

    std::map<primitive_id, std::pair<format::type, impl_types>> _forcing_map;
    static const std::vector<std::pair<format::type, bool>> optimized_formats;  // pair of format type and allowed weak restriction
    size_t _total_conv;
    std::map<std::pair<format::type, bool>, size_t> _optimized_conv_count;

    layout get_expected_layout(layout const& current_layout,
                               convolution_node const& node,
                               layout const& output_or_weights_layout);
    layout get_expected_layout(layout const& current_layout,
                               deconvolution_node const& node,
                               layout const& output_or_weights_layout);
    layout get_expected_layout(layout const& current_layout,
                               detection_output_node const& node,
                               layout const& output_or_weights_layout);
    layout get_expected_layout(layout const& current_layout,
                               binary_convolution_node const& node,
                               layout const& output_or_weights_layout);

    bool is_depthwise(const convolution_node& node) const;
    format imad_case(convolution_node const& node) const;

    bool convolution_bfyx_opt(const layout& output_layout,
                              const layout& weights_layout,
                              std::shared_ptr<const convolution> conv);
    bool convolution_byxf_opt(const layout& input_layout,
                              const layout& output_layout,
                              const layout& weights_layout,
                              const convolution_node& node);
    bool convolution_b_fs_yx_fsv16_opt(const layout& input_layout,
                                       const layout& output_layout,
                                       const layout& weights_layout,
                                       std::shared_ptr<const convolution> conv,
                                       bool weak_restrictions = false);
    bool convolution_b_fs_zyx_fsv16_opt(const layout& input_layout,
                                        const layout& output_layout,
                                        const layout& weights_layout,
                                        std::shared_ptr<const convolution> conv);
    bool convolution_bs_fs_yx_bsv16_fsv16_opt(const layout& input_layout,
                                              const layout& output_layout,
                                              const layout& weights_layout,
                                              std::shared_ptr<const convolution> conv);
    bool convolution_fs_b_yx_fsv32_opt(const layout& input_layout,
                                       const layout& output_layout,
                                       const layout& weights_layout,
                                       std::shared_ptr<const convolution> conv,
                                       bool weak_restrictions = false);
    bool deconvolution_b_fs_zyx_fsv16_opt(const layout &input_layout,
                                          const layout &weights_layout,
                                          std::shared_ptr<const deconvolution> conv);
    bool deconvolution_b_fs_yx_fsv16_opt(const layout &input_layout,
                                         const layout &weights_layout,
                                         std::shared_ptr<const deconvolution> conv);
    bool users_for_convolution_byxf_opt(program_node const& node, uint32_t depth);
    bool deps_for_convolution_byxf_opt(program_node const& node, uint32_t depth);

public:
    explicit layout_optimizer(bool output_size_handling_enabled = true);

    format get_preferred_format(program_node& node);
    impl_types get_preferred_impl_type(program_node& node);

    bool is_format_supported(program_node& node, format::type fmt);

    // Returns whether reorder between "prev" with format fmt_prev and "next" with format fmt_next
    // can be fused into next.
    bool can_fuse_reorder(program_node& prev, program_node& next, format fmt_prev, format fmt_next);
    bool can_fuse_reorder_to_prev(program_node& prev, program_node& next, format fmt_prev, format fmt_next);

    void set_optimization_attribute(optimization_attributes_type attribute, int32_t val);
    optimization_attributes get_optimization_attributes() { return _optimization_attributes; }

    void set_implementation_forcing(const implementation_forcing_map& map);

    void update_formats_map(const convolution_node& node);
    bool is_format_optimized(const convolution_node& node, const format& format, bool use_weak_restrictions = false);
    bool is_format_optimized(const deconvolution_node& node, const format& format);
    size_t get_optimized_conv_count(const std::pair<format::type, bool>& format);
    size_t get_total_conv_count();

    bool should_select_b_fs_yx_fsv16_layout(convolution_node const& node, layout const& output_or_weights_layout);
};
}  // namespace cldnn
