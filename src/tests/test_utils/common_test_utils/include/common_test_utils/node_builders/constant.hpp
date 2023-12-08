// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/node_builders/broadcast.hpp"
#include "openvino/core/node.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace test {
namespace utils {
template <class T>
std::shared_ptr<Node> make_constant(const element::Type& type, const Shape& shape, const T& num) {
    std::shared_ptr<Node> val = nullptr;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
    std::string unsupported_data_type;
    switch (type) {
    case element::Type_t::f32:
        val =
            std::make_shared<ov::op::v0::Constant>(type, ov::Shape{}, std::vector<float>{static_cast<float>(num)});
        break;
    case element::Type_t::f64:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<double>{static_cast<double>(num)});
        break;
    case element::Type_t::f16:
        val = std::make_shared<ov::op::v0::Constant>(
            type,
            ov::Shape{},
            std::vector<ov::float16>{ov::float16(static_cast<float>(num))});
        break;
    case element::Type_t::bf16:
        val = std::make_shared<ov::op::v0::Constant>(
            type,
            ov::Shape{},
            std::vector<ov::bfloat16>{ov::bfloat16(static_cast<float>(num))});
        break;
    case element::Type_t::i64:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<int64_t>{static_cast<int64_t>(num)});
        break;
    case element::Type_t::i32:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<int32_t>{static_cast<int32_t>(num)});
        break;
    case element::Type_t::i16:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<int16_t>{static_cast<int16_t>(num)});
        break;
    case element::Type_t::i8:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<int8_t>{static_cast<int8_t>(num)});
        break;
    case element::Type_t::u64:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<uint64_t>{static_cast<uint64_t>(num)});
        break;
    case element::Type_t::u32:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<uint32_t>{static_cast<uint32_t>(num)});
        break;
    case element::Type_t::u16:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<uint16_t>{static_cast<uint16_t>(num)});
        break;
    case element::Type_t::u8:
        val = std::make_shared<ov::op::v0::Constant>(type,
                                                     ov::Shape{},
                                                     std::vector<uint8_t>{static_cast<uint8_t>(num)});
        break;
    case element::Type_t::dynamic:
        unsupported_data_type = "dynamic";
        break;
    case element::Type_t::boolean:
        unsupported_data_type = "boolean";
        break;
    case element::Type_t::u1:
        unsupported_data_type = "u1";
        break;
    case element::Type_t::i4:
        unsupported_data_type = "i4";
        break;
    case element::Type_t::u4:
        unsupported_data_type = "u4";
        break;
    case element::Type_t::nf4:
        unsupported_data_type = "nf4";
        break;
    case element::Type_t::string:
        unsupported_data_type = "string";
        break;
    case element::Type_t::undefined:
        unsupported_data_type = "undefined";
        break;
    }
    if (!unsupported_data_type.empty())
        OPENVINO_THROW("make_constant: Unsupported element type '", unsupported_data_type, "'");

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif

    if (shape.size() > 0) {
        ov::AxisSet axes;
        for (size_t i = 0; i < shape.size(); i++) {
            axes.insert(i);
        }
        val = builder::opset1::make_broadcast(val, shape, axes).get_node_shared_ptr();
    }

    return val;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
