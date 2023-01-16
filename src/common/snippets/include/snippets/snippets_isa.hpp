// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ops.hpp"
#include <ngraph/opsets/opset1.hpp>

#include "op/broadcastload.hpp"
#include "op/broadcastmove.hpp"
#include "op/convert_saturation.hpp"
#include "op/convert_truncation.hpp"
#include "op/kernel.hpp"
#include "op/load.hpp"
#include "op/nop.hpp"
#include "op/scalar.hpp"
#include "op/powerstatic.hpp"
#include "op/store.hpp"
#include "op/tile.hpp"
#include "op/tile_scheduler.hpp"

namespace ngraph {
namespace snippets {
namespace isa {
#define NGRAPH_OP(a, b) using b::a;
#include "snippets_isa_tbl.hpp"
#undef NGRAPH_OP
} // namespace isa
} // namespace snippets
} // namespace ngraph
