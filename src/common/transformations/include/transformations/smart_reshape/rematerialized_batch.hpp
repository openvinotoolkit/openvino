// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "openvino/core/model.hpp"
#include "transformations_visibility.hpp"

namespace ov {

/**
 * @brief Returns the leading dimension the model is locked to when some Reshape rebuilds the batch of the given input
 * out of a statically known leading dimension.
 *
 * Propagates a symbol seeded on the leading dimension of the input and looks for a Reshape whose data keeps a static
 * leading dimension while its result carries that symbol. Such a Reshape takes the batch from a shape expression
 * instead of from its data, which means the batch was pinned earlier and only the value the model was converted with
 * produces correct results. Returns no value when the batch reaches the model outputs through the data.
 *
 * Symbols relate dimensions that stay equal, so a batch fused into a product is not tracked. The check reports a
 * proven inconsistency, not that reshaping is safe.
 *
 * @param model Model to analyze. It is not modified.
 * @param input_index Index of the input whose leading dimension holds the batch.
 */
TRANSFORMATIONS_API std::optional<int64_t> find_rematerialized_batch(const Model& model, size_t input_index);

}  // namespace ov
