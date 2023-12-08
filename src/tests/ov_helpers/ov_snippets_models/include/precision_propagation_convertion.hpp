// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/model.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

/**
 * @class PrecisionPropagationConvertionFunction
 * @brief PrecisionPropagationConvertionFunction instance returns reference and original functions.
 *
 * Input arguments are used to create function in getOriginal methods only.
 * Dont use getReference and getLowered method, they are not implemented and throw std::runtime_error exception.
 * Note, ov::element::Type_t precision base type input argument is not used.
 */
class PrecisionPropagationConvertionFunction : public SnippetsFunctionBase {
public:
    PrecisionPropagationConvertionFunction(
        const std::vector<ov::PartialShape>& input_shapes,
        const element::Type input_type,
        const std::vector<float>& fake_quantize_intervals);

protected:
    std::shared_ptr<Model> initOriginal() const override;

private:
    /*
     * Returns model implicitly via getOriginal call in initOriginal.
     */
    static std::shared_ptr<ov::Model> get(
        const std::vector<ov::PartialShape>& input_shapes,
        const element::Type input_type,
        const std::vector<float>& fake_quantize_intervals);

    const std::vector<float> fake_quantize_intervals;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
