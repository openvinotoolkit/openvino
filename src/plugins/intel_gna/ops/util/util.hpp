// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset8.hpp>
#include <vector>
#include <memory>

namespace GNAPluginNS {
namespace ngraph_util {

template <typename T>
static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    using A = typename ov::element_type_traits<T::value>::value_type;
    const auto& v = constant->get_vector<A>();
    std::copy(v.begin(), v.end(), std::back_inserter(values));
    return true;
}

static bool get_constant_value(std::tuple<>&&, const std::shared_ptr<ngraph::opset8::Constant>&, std::vector<double>&) {
    return false;
}

template<typename T, typename ...Types>
static bool get_constant_value(std::tuple<T, Types...>&&,
                  const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    return constant->get_element_type() == T::value &&
           get_constant_value<T>(constant, values) ||
           get_constant_value(std::tuple<Types...>(), constant, values);
}

static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, std::vector<double>& values) {
    return get_constant_value(std::tuple<std::integral_constant<ov::element::Type_t, ov::element::i32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::i64>,
                                         std::integral_constant<ov::element::Type_t, ov::element::u32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::u64>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f16>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f32>,
                                         std::integral_constant<ov::element::Type_t, ov::element::f64>>(),
                              constant,
                              values);
}

static bool get_constant_value(const std::shared_ptr<ngraph::opset8::Constant>& constant, double& value) {
    std::vector<double> values;
    if (!get_constant_value(constant, values)) {
        return false;
    }

    if (values.empty() || values.size() > 1) {
        throw std::runtime_error("The size of values is more than 1.");
    }

    value = values[0];
    return true;
}

} // namespace ngraph_util
} // namespace GNAPluginNS
