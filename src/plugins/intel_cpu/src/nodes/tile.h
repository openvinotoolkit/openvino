// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "common/tile_broadcast_utils.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Tile : public Node, public TileBroadcastCommon {
public:
    Tile(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;

private:
    void plainExecute(const dnnl::stream& strm);

    static constexpr size_t TILE_INPUT = 0LU;
    static constexpr size_t TILE_REPEATS = 1LU;

    int axis = -1;
    int tiles = 0;
    bool noTiling = false;
    VectorDims originRepeats;
};

}  // namespace ov::intel_cpu::node
