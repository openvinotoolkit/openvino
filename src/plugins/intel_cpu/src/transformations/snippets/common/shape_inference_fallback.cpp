// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

namespace ov::snippets {

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry =
    detail::make_common_cpu_shape_infer_registry();

}  // namespace ov::snippets
