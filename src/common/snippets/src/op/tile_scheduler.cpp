// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile_scheduler.hpp"
#include "snippets/generator.hpp"

ngraph::snippets::op::TileScheduler::TileScheduler(const std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> &vector_region,
                                                   const std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> &scalar_region)
    : Op(), vector_region{vector_region}, scalar_region{scalar_region} {
}
