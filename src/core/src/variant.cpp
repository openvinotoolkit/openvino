// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

namespace ngraph {
OPENVINO_SUPPRESS_DEPRECATED_START
template class VariantImpl<std::string>;
template class VariantImpl<int64_t>;
template class VariantImpl<bool>;
OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace ngraph
