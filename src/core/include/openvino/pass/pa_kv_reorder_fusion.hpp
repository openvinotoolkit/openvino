// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

/// Fuses paired key/value cache reorder paths into a single PaKVReorder op.
///
/// This transformation performs graph pattern matching and replacement.
/// An optional cache_precision parameter can be provided to set the element type
/// of the key/value cache Parameters during fusion.
class OPENVINO_API PaKVReorderFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PaKVReorderFusion");
    PaKVReorderFusion() = default;
    explicit PaKVReorderFusion(ov::element::Type cache_precision);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    ov::element::Type m_cache_precision = ov::element::dynamic;
};

}  // namespace pass
}  // namespace ov
