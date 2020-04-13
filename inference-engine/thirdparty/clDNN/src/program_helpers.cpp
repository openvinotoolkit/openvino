/*
// Copyright (c) 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_helpers.h"
#include "program_impl.h"
#include "data_inst.h"
#include <algorithm>
#include <utility>

namespace cldnn {
// helper function for merging the weights/biases buffers on cpu side for depthwise separable convolution optimization
void program_helpers::merge_buffers(engine_impl& engine,
                                    program_node& node,
                                    const layout& target_layout,
                                    size_t begin_offset,
                                    size_t end_offset) {
    memory_impl::ptr data_to_allocate = engine.allocate_memory(target_layout, 0, false);

    for (size_t i = begin_offset; i < end_offset; i++) {
        auto& weights = node.get_dependency(i).as<data>();
        mem_lock<char> src{weights.get_attached_memory()};
        mem_lock<char> dst{data_to_allocate};
        std::copy(src.begin(), src.end(), dst.begin() + (i - begin_offset) * src.size());
    }

    for (size_t i = 0; i < end_offset - begin_offset - 1; i++) node.remove_dependency(begin_offset + 1);

    auto& data_node = node.get_dependency(begin_offset).as<data>();
    data_node.attach_memory(*data_to_allocate, false);
}

// helper function for getting target layout used in depthwise sep optimization
layout program_helpers::get_weights_layout(typed_program_node<cldnn::data>& data_node, int32_t split) {
    auto mem_layout = data_node.get_output_layout();

    return layout(mem_layout.data_type,
                  mem_layout.format,
                  {split * mem_layout.size.batch[0],
                   mem_layout.size.feature[0],
                   mem_layout.size.spatial[0],
                   mem_layout.size.spatial[1]});
}

// pair.first tells whether l1 and l2 are absolutely identical
// pair.second tells whether l1 and l2 can be reinterpreted to each other without need of reordering
// note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted
// nor dropped) note: if layouts describe two buffers with different size, consider them not to be identical even if
// smaller buffer can be considered to hold subsequence of larger buffer,
//       this behavior is required to force buffer allocation for smaller buffer which, currently, should always be
//       performed
std::pair<bool, bool> program_helpers::are_layouts_identical(layout const& l1, layout const& l2) {
    const auto& l1_pad = l1.data_padding;
    const auto& l2_pad = l2.data_padding;
    if (l1 == l2)
        return {true, true};
    if (l1.data_type != l2.data_type)
        return {false, false};
    // Reorders between bfyx, bfzyx, bfwzyx can pe reinterpeted as reshape when
    // there is no padding and both hold same number of elements.
    if ((l1.format == format::bfyx || l1.format == format::bfzyx || l1.format == format::bfwzyx) &&
        (l2.format == format::bfyx || l2.format == format::bfzyx || l2.format == format::bfwzyx) && !l1_pad &&
        !l2_pad && l1.get_linear_size() == l2.get_linear_size())
        return {false, true};
    if (l1.size != l2.size)
        return {false, false};
    if (l1.get_linear_size() != l2.get_linear_size())
        return {false, false};
    if ((l1.format == format::bf8_xy16 && l2.format != format::bf8_xy16) ||
        (l2.format == format::bf8_xy16 && l1.format != format::bf8_xy16) ||
        (l1.format == format::b_fs_yx_fsv4 && l2.format != format::b_fs_yx_fsv4) ||
        (l2.format == format::b_fs_yx_fsv4 && l1.format != format::b_fs_yx_fsv4) ||
        (l1.format == format::fs_b_yx_fsv32 && l2.format != format::fs_b_yx_fsv32) ||
        (l2.format == format::fs_b_yx_fsv32 && l1.format != format::fs_b_yx_fsv32) ||
        (l1.format == format::b_fs_yx_fsv16 && l2.format != format::b_fs_yx_fsv16) ||
        (l2.format == format::b_fs_yx_fsv16 && l1.format != format::b_fs_yx_fsv16) ||
        (l1.format == format::b_fs_yx_fsv32 && l2.format != format::b_fs_yx_fsv32) ||
        (l2.format == format::b_fs_yx_fsv32 && l1.format != format::b_fs_yx_fsv32) ||
        (l1.format == format::b_fs_zyx_fsv32 && l2.format != format::b_fs_zyx_fsv32) ||
        (l2.format == format::b_fs_zyx_fsv32 && l1.format != format::b_fs_zyx_fsv32) ||
        (l1.format == format::b_fs_zyx_fsv16 && l2.format != format::b_fs_zyx_fsv16) ||
        (l2.format == format::b_fs_zyx_fsv16 && l1.format != format::b_fs_zyx_fsv16) ||
        (l1.format == format::bs_fs_yx_bsv16_fsv16 && l2.format != format::bs_fs_yx_bsv16_fsv16) ||
        (l2.format == format::bs_fs_yx_bsv16_fsv16 && l1.format != format::bs_fs_yx_bsv16_fsv16) ||
        (l1.format == format::bs_fs_zyx_bsv16_fsv16 && l2.format != format::bs_fs_zyx_bsv16_fsv16) ||
        (l2.format == format::bs_fs_zyx_bsv16_fsv16 && l1.format != format::bs_fs_zyx_bsv16_fsv16))
        return {false, false};

    auto l1_pitch = l1.get_pitches();
    auto l2_pitch = l2.get_pitches();

    // ignore pitches which will never be used (for dims with size == 1)
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l1.size.raw[i] == 1)
            l1_pitch.raw[i] = 0;
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l2.size.raw[i] == 1)
            l2_pitch.raw[i] = 0;

    auto l1_offset = l1.get_linear_offset();
    auto l2_offset = l2.get_linear_offset();
    if (l1_pitch == l2_pitch && l1_offset == l2_offset)
        return {false, true};

    return {false, false};
}
}  // namespace cldnn
