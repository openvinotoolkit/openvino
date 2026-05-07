// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "../v1/subgraph_pipeline.hpp"

namespace ov {
namespace npuw {
namespace compiled {
struct Attention;
struct PyramidAttention;
struct HostFlashAttention;
}  // namespace compiled
namespace s11n {
class Stream;
struct SubmodelDeserializeCtx;
}  // namespace s11n
namespace attn {

enum class BehaviorKind { Dynamic, Pyramid, HFA };

using CompiledDynamicState = std::shared_ptr<ov::npuw::compiled::Attention>;
using CompiledPyramidState = std::shared_ptr<ov::npuw::compiled::PyramidAttention>;
using CompiledHFAState = std::shared_ptr<ov::npuw::compiled::HostFlashAttention>;

void put_compiled_dynamic(v1::subgraphs::Context& context, CompiledDynamicState state);
void put_compiled_pyramid(v1::subgraphs::Context& context, CompiledPyramidState state);
void put_compiled_hfa(v1::subgraphs::Context& context, CompiledHFAState state);

ov::npuw::compiled::Attention* get_compiled_dynamic(v1::subgraphs::Context& context);
const ov::npuw::compiled::Attention* get_compiled_dynamic(const v1::subgraphs::Context& context);

ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(v1::subgraphs::Context& context);
const ov::npuw::compiled::PyramidAttention* get_compiled_pyramid(const v1::subgraphs::Context& context);

ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(v1::subgraphs::Context& context);
const ov::npuw::compiled::HostFlashAttention* get_compiled_hfa(const v1::subgraphs::Context& context);

bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline);

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx);

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry);

void attach_runtime_behavior(ov::npuw::v1::subgraphs::CompiledPipeline& compiled_pipeline,
                             ov::npuw::v1::subgraphs::Context& compiled_context,
                             BehaviorKind kind);

}  // namespace attn
}  // namespace npuw
}  // namespace ov
