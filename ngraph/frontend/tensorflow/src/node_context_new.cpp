// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context_new.hpp"

#define NGRAPH_VARIANT_DEFINITION(TYPE)                        \
    constexpr VariantTypeInfo VariantWrapper<TYPE>::type_info; \
    template class ngraph::VariantImpl<TYPE>;

namespace ov {
NGRAPH_VARIANT_DEFINITION(int32_t)
NGRAPH_VARIANT_DEFINITION(std::vector<int32_t>)
NGRAPH_VARIANT_DEFINITION(float)
NGRAPH_VARIANT_DEFINITION(std::vector<float>)
NGRAPH_VARIANT_DEFINITION(bool)
NGRAPH_VARIANT_DEFINITION(ngraph::PartialShape)
NGRAPH_VARIANT_DEFINITION(ov::element::Type)
NGRAPH_VARIANT_DEFINITION(std::vector<int64_t>)
NGRAPH_VARIANT_DEFINITION(std::vector<std::string>)
NGRAPH_VARIANT_DEFINITION(::tensorflow::DataType)
NGRAPH_VARIANT_DEFINITION(::tensorflow::TensorProto)
}  // namespace ov

/*
namespace ngraph {
namespace frontend {
namespace tf {
#define TEMPLATE_EXPLICIT_SPECIALIZATION(T)                                \
    template T NodeContext::get_attribute<T>(const std::string&) const; \
    template T NodeContext::get_attribute<T>(const std::string&, const T&) const;

#define TEMPLATE_EXPLICIT_SPECIALIZATION_V(T)                                                                        \
    template std::vector<T> NodeContext::get_attribute<std::vector<T>>(const std::string&) const;                 \
    template std::vector<T> NodeContext::get_attribute<std::vector<T>>(const std::string&, const std::vector<T>&) \
        const;

TEMPLATE_EXPLICIT_SPECIALIZATION_V(int32_t)
TEMPLATE_EXPLICIT_SPECIALIZATION_V(int64_t)
TEMPLATE_EXPLICIT_SPECIALIZATION_V(float)

TEMPLATE_EXPLICIT_SPECIALIZATION(int32_t)
TEMPLATE_EXPLICIT_SPECIALIZATION(int64_t)
TEMPLATE_EXPLICIT_SPECIALIZATION(ngraph::element::Type)
TEMPLATE_EXPLICIT_SPECIALIZATION(ngraph::PartialShape)
TEMPLATE_EXPLICIT_SPECIALIZATION(std::string)
TEMPLATE_EXPLICIT_SPECIALIZATION(bool)
TEMPLATE_EXPLICIT_SPECIALIZATION(float)
TEMPLATE_EXPLICIT_SPECIALIZATION_V(std::string)
}
}  // namespace frontend
}  // namespace ngraph
*/