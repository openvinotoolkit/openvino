// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "storeconvert.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface ScalarStore
 * @brief Generated for store and convert a scalar value at the same time
 * @ingroup snippets
 */
class ScalarStoreConvert : public StoreConvert {
public:
    OPENVINO_OP("ScalarStoreConvert", "SnippetsOpset", ngraph::snippets::op::StoreConvert);

    ScalarStoreConvert(const Output<Node>& x, const ov::element::Type& destination_type);
    ScalarStoreConvert() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarStoreConvert>(new_args.at(0), m_destination_type);
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
