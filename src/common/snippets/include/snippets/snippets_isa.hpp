// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op/brgemm.hpp"
#include "op/broadcastload.hpp"
#include "op/broadcastmove.hpp"
#include "op/buffer.hpp"
#include "op/convert_saturation.hpp"
#include "op/convert_truncation.hpp"
#include "op/fill.hpp"
#include "op/horizon_max.hpp"
#include "op/horizon_sum.hpp"
#include "op/kernel.hpp"
#include "op/load.hpp"
#include "op/loop.hpp"
#include "op/nop.hpp"
#include "op/perf_count.hpp"
#include "op/powerstatic.hpp"
#include "op/rank_normalization.hpp"
#include "op/reduce.hpp"
#include "op/reg_spill.hpp"
#include "op/reorder.hpp"
#include "op/reshape.hpp"
#include "op/scalar.hpp"
#include "op/store.hpp"
#include "op/vector_buffer.hpp"
#include "openvino/core/node.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov::snippets::isa {
#define OV_OP(a, b) using b::a;
#include "snippets_isa_tbl.hpp"
#undef OV_OP
}  // namespace ov::snippets::isa
