// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/make_constant.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace builder {
std::shared_ptr<Node> make_constant_from_double(const element::Type& type, const Shape& shape, double num) {
    auto ceil_func = [](double x) {
        return ceil(x);
    };

    std::shared_ptr<ngraph::Node> result = nullptr;
    switch (type) {
    case element::Type_t::i8: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<int8_t>(num, ceil_func));
        break;
    }
    case element::Type_t::i16: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<int16_t>(num, ceil_func));
        break;
    }
    case element::Type_t::i32: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<int32_t>(num, ceil_func));
        break;
    }
    case element::Type_t::i64: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<int64_t>(num, ceil_func));
        break;
    }
    case element::Type_t::u8: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<uint8_t>(num, ceil_func));
        break;
    }
    case element::Type_t::u16: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<uint16_t>(num, ceil_func));
        break;
    }
    case element::Type_t::u32: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<uint32_t>(num, ceil_func));
        break;
    }
    case element::Type_t::u64: {
        result = std::make_shared<ngraph::op::Constant>(type, shape, double_to_int<uint64_t>(num, ceil_func));
        break;
    }
    case element::Type_t::f16: {
        result = builder::make_constant(type, shape, static_cast<float16>(num));
        break;
    }
    case element::Type_t::bf16: {
        result = builder::make_constant(type, shape, static_cast<bfloat16>(num));
        break;
    }
    case element::Type_t::f32: {
        result = builder::make_constant(type, shape, static_cast<float>(num));
        break;
    }
    case element::Type_t::f64: {
        result = builder::make_constant(type, shape, num);
        break;
    }
    default:
        OPENVINO_THROW("Unsupported data type during make_constant_from_double");
        break;
    }
    return result;
}
}  // namespace builder
}  // namespace ngraph
