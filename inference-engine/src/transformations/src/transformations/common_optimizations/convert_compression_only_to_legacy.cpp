// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"

#include "transformations/convert_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

bool ov::pass::ConvertCompressedOnlyToLegacy::run_on_function(std::shared_ptr<ov::Function> f) {
    if (ngraph::op::util::has_decompression_converts(f)) {
        const precisions_array convert_precision_list{
            {ov::element::f32, ov::element::f16}
        };
        auto convert_precision = ngraph::pass::ConvertPrecision(convert_precision_list);
        return convert_precision.run_on_function(f);
    }
    return false;
}
