// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace test {
namespace utils {

namespace deprecated {
// Legacy implementation
// Remove after transition to new one
template <typename T>
std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const std::vector<size_t>& shape,
                                        const std::vector<T>& data,
                                        bool random = false,
                                        T up_to = 10,
                                        T start_from = 1,
                                        const int seed = 1) {
#define makeNode(TYPE)                                                                                              \
    case TYPE:                                                                                                      \
        if (random) {                                                                                               \
            return std::make_shared<ov::op::v0::Constant>(                                                          \
                type,                                                                                               \
                shape,                                                                                              \
                NGraphFunctions::Utils::generateVector<TYPE>(ov::shape_size(shape),                                 \
                                                             ov::element_type_traits<TYPE>::value_type(up_to),      \
                                                             ov::element_type_traits<TYPE>::value_type(start_from), \
                                                             seed));                                                \
        } else {                                                                                                    \
            if (std::is_same<T, fundamental_type_for<TYPE>>::value) {                                               \
                return std::make_shared<ov::op::v0::Constant>(type, shape, data);                                   \
            } else {                                                                                                \
                /* Convert std::vector<T> data to required type */                                                  \
                std::vector<fundamental_type_for<TYPE>> converted_data(data.size());                                \
                std::transform(data.cbegin(), data.cend(), converted_data.begin(), [](T e) {                        \
                    return static_cast<fundamental_type_for<TYPE>>(e);                                              \
                });                                                                                                 \
                return std::make_shared<ov::op::v0::Constant>(type, shape, converted_data);                         \
            }                                                                                                       \
        }                                                                                                           \
        break;
    switch (type) {
        makeNode(ov::element::bf16);
        makeNode(ov::element::f16);
        makeNode(ov::element::f32);
        makeNode(ov::element::f64);
        makeNode(ov::element::i8);
        makeNode(ov::element::i16);
        makeNode(ov::element::i32);
        makeNode(ov::element::i64);
        makeNode(ov::element::u8);
        makeNode(ov::element::u16);
        makeNode(ov::element::u32);
        makeNode(ov::element::u64);
        makeNode(ov::element::boolean);
        makeNode(ov::element::nf4);
        makeNode(ov::element::u4);
        makeNode(ov::element::i4);
    default:
        throw std::runtime_error("Unhandled precision");
    }
#undef makeNode
}
}  // namespace deprecated

std::shared_ptr<ov::Node> make_constant(const ov::element::Type& type,
                                        const ov::Shape& shape,
                                        const InputGenerateData& in_data = InputGenerateData(1, 9, 1, 1));

}  // namespace utils
}  // namespace test
}  // namespace ov
