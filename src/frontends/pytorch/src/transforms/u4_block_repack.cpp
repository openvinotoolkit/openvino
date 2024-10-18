// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "u4_block_repack.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
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

U4BlockRepack::U4BlockRepack(bool is_symmetrical) {
    const auto& m_constant = wrap_type<v0::Constant>();
    const auto& m_reshape1 = wrap_type<v1::Reshape>({m_constant, any_input()});
    const auto& m_transpose = wrap_type<v1::Transpose>({m_reshape1, any_input()});
    const auto& m_reshape2 = wrap_type<v1::Reshape>({m_transpose, any_input()});

    auto pack_byte = [](uint8_t lo, uint8_t hi) -> uint8_t {
        return (hi << 4) | (lo & 0x0F);
    };  // swap halfs because Convert op assumes this layout

    const std::function<uint8_t(const uint8_t*, size_t)>& get_u4 = [](const uint8_t* src, size_t idx) {
        const size_t byte_idx = idx / 2;
        const uint8_t bit_shift = 4 * (idx % 2);
        return (src[byte_idx] >> bit_shift) & 0xF;
    };

    const std::function<uint8_t(const uint8_t*, size_t)>& get_i4 = [get_u4](const uint8_t* src, size_t idx) {
        // by flipping first bit we get same effect as subtracting 8
        return get_u4(src, idx) ^ 0b1000;
    };

    register_matcher(
        std::make_shared<Matcher>(m_reshape2, "ov::frontend::pytorch::pass::U4BlockRepack"),
        [=](Matcher& m) {
            auto& pattern_to_output = m.get_pattern_value_map();
            auto constant =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output[m_constant].get_node_shared_ptr());
            if (!constant)
                return false;
            auto reshape1 = pattern_to_output[m_reshape1].get_node_shared_ptr();
            auto transpose = pattern_to_output[m_transpose].get_node_shared_ptr();
            auto reshape2 = pattern_to_output[m_reshape2].get_node_shared_ptr();
            auto pattern_root = reshape2;

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

            auto get_number = get_u4;
            auto constant_dtype = element::u4;
            NodeVector copy_from{std::move(constant), std::move(reshape1), std::move(transpose), reshape2};
            if (is_symmetrical) {
                get_number = get_i4;
                constant_dtype = element::i4;
                // find pattern Convert(W, i8) -> Subtract(8)
                auto reshape_targets = reshape2->output(0).get_target_inputs();
                if (reshape_targets.size() != 1)
                    return false;
                auto convert = reshape_targets.begin()->get_node()->shared_from_this();
                if (!std::dynamic_pointer_cast<ov::op::v0::Convert>(convert))
                    return false;
                auto convert_targets = convert->output(0).get_target_inputs();
                if (convert_targets.size() != 1)
                    return false;
                auto subtract = convert_targets.begin()->get_node()->shared_from_this();
                if (!std::dynamic_pointer_cast<ov::op::v1::Subtract>(subtract))
                    return false;
                pattern_root = subtract;
                copy_from.push_back(std::move(convert));
                copy_from.push_back(subtract);
            }
            auto new_const = std::make_shared<v0::Constant>(constant_dtype, destination_shape);
            auto dst = const_cast<uint8_t*>(                                   // const_cast?
                reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));  // TODO: How to better access u4 data?

            for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
                auto src_block = src + iblock * block_size;
                auto dst_block = dst + iblock * block_size;
                for (size_t i = 0; i < lane_size; ++i) {
                    for (size_t j = 0; j < block_height / 2; ++j) {  // /2 because we handle two bytes at once
                        uint8_t lo = get_number(src_block, 2 * j * lane_size + i);
                        uint8_t hi = get_number(src_block, (2 * j + 1) * lane_size + i);
                        dst_block[i * block_height / 2 + j] = pack_byte(lo, hi);
                    }
                }
            }

            copy_runtime_info(copy_from, new_const);
            replace_node(pattern_root, new_const);

            return true;
        });
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
