// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkPrecisionSensitiveSubgraphs;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkPrecisionSensitiveSubgraphs transformation goes up and marks nodes
 * inside the subgraph starting from precision-sensitive input and ending at
 * the ShapeOf node as disabled for FP16 compression.
 *
 * If mark_only_consts is true only Consts are marked. False value is used
 * during offline_transformations. For the GPU plugin need to keep
 * the whole shape subgraph in FP32, therefore mark_only_consts need to be set false for GPU.
 */
class ov::pass::MarkPrecisionSensitiveSubgraphs : public ModelPass {
public:
    OPENVINO_RTTI("MarkPrecisionSensitiveSubgraphs", "0");
    explicit MarkPrecisionSensitiveSubgraphs(bool mark_only_consts = true) : m_mark_only_consts(mark_only_consts) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    bool m_mark_only_consts = true;
};
