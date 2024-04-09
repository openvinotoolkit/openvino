// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/opsets/opset1.hpp"

#include "op/broadcastload.hpp"
#include "op/broadcastmove.hpp"
#include "op/buffer.hpp"
#include "op/convert_saturation.hpp"
#include "op/convert_truncation.hpp"
#include "op/horizon_max.hpp"
#include "op/horizon_sum.hpp"
#include "op/fill.hpp"
#include "op/kernel.hpp"
#include "op/load.hpp"
#include "op/reshape.hpp"
#include "op/nop.hpp"
#include "op/scalar.hpp"
#include "op/powerstatic.hpp"
#include "op/store.hpp"
#include "op/loop.hpp"
#include "op/brgemm.hpp"
#include "op/vector_buffer.hpp"
#include "op/rank_normalization.hpp"
#include "op/perf_count.hpp"
#include "op/reduce.hpp"

namespace ov {
namespace snippets {
namespace isa {
#define OV_OP(a, b) using b::a;
#include "snippets_isa_tbl.hpp"
#undef OV_OP
} // namespace isa
} // namespace snippets
} // namespace ov
