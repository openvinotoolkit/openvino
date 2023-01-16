// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile_scheduler.hpp"
#include "snippets/generator.hpp"

ngraph::snippets::op::TileScheduler::TileScheduler(const AllocatedEmitter& vector_region, const AllocatedEmitter& scalar_region)
    : Op(), vector_region{vector_region}, scalar_region{scalar_region} {
}
