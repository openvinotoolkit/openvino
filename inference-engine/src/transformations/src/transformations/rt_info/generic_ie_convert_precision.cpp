// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <functional>
#include <memory>
#include <iterator>
#include <ostream>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include "transformations/rt_info/generic_ie_convert_precision.hpp"

namespace ngraph {

template <typename T>
VariantImpl<T>::~VariantImpl() { }

template class ngraph::VariantImpl<GenericIEConvertPrecision>;

constexpr VariantTypeInfo VariantWrapper<GenericIEConvertPrecision>::type_info;

}  // namespace ngraph