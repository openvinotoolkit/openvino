// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "u4_block_repack.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

U4BlockRepack::U4BlockRepack() {
    const auto m_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto m_reshape1 = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({m_constant, any_input()});
    const auto m_transpose = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({m_reshape1, any_input()});
    const auto m_reshape2 = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({m_transpose, any_input()});

    auto pack_byte = [](uint8_t lo, uint8_t hi) {
        return (hi << 4) | (lo & 0x0F);
    };  // swap halfs because Convert op assumes this layout

    auto get_u4 = [](const uint8_t* src, size_t idx) {
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (idx % 2);
        return (src[byte_idx] >> bit_shift) & 0xF;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(m_reshape2, "ov::frontend::pytorch::pass::U4BlockRepack"),
        [=](ov::pass::pattern::Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto constant =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output[m_constant].get_node_shared_ptr());
            if (!constant)
                return false;
            auto reshape1 = pattern_to_output[m_reshape1].get_node_shared_ptr();
            auto transpose = pattern_to_output[m_transpose].get_node_shared_ptr();
            auto reshape2 = pattern_to_output[m_reshape2].get_node_shared_ptr();

            if (constant->get_element_type() != element::u4)
                return false;

            // FIXME: Check reshape/transpose/reshape target shapes and axes permutation; now they are supposed to be
            // always in expected form

            auto source_shape = reshape1->get_output_shape(0);

            if (source_shape.size() != 3)
                return false;

            auto destination_shape = reshape2->get_output_shape(0);

            size_t n_blocks = source_shape[0];
            size_t block_height = source_shape[1];
            size_t lane_size = source_shape[2];                // size in u4 units
            size_t block_size = block_height * lane_size / 2;  // size in bytes

            auto src = constant->get_data_ptr<uint8_t>();

            auto new_const = std::make_shared<v0::Constant>(element::u4, destination_shape);
            auto dst = const_cast<uint8_t*>(                                   // const_cast?
                reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));  // TODO: How to better accees u4 data?

            for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
                auto src_block = src + iblock * block_size;
                auto dst_block = dst + iblock * block_size;
                for (size_t i = 0; i < lane_size; ++i) {
                    for (size_t j = 0; j < block_height / 2; ++j) {  // /2 because we handle two bytes at once
                        uint8_t lo = get_u4(src_block, 2 * j * lane_size + i);
                        uint8_t hi = get_u4(src_block, (2 * j + 1) * lane_size + i);
                        dst_block[i * block_height / 2 + j] = pack_byte(lo, hi);
                    }
                }
            }

            copy_runtime_info(NodeVector{constant, reshape1, transpose, reshape2}, new_const);
            replace_node(reshape2, new_const);

            return true;
        });
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
