// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class BatchToSpaceFunction {
public:
    static std::shared_ptr<ov::Model> get(const ov::PartialShape& input_shape,
                                                 const ov::element::Type input_type,
                                                 const FakeQuantizeOnData& fq_on_data,
                                                 const std::vector<size_t>& block_shape,
                                                 const std::vector<size_t>& crops_begin,
                                                 const std::vector<size_t>& crops_end);

    static std::shared_ptr<ov::Model> get(const ov::PartialShape& input_shape,
                                                 const ov::element::Type input_type,
                                                 const ngraph::builder::subgraph::DequantizationOperations& dequantization_before,
                                                 const std::vector<size_t>& block_shape,
                                                 const std::vector<size_t>& crops_begin,
                                                 const std::vector<size_t>& crops_end,
                                                 const ngraph::builder::subgraph::DequantizationOperations& dequantization_after = {});
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
