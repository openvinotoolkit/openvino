// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "../v1/subgraph_pipeline.hpp"

namespace ov {
class Model;
namespace npuw {
namespace function {
struct MoEExperts;
struct MoEDownstream;
}  // namespace function
namespace compiled {
struct MoEExperts;
struct MoEDownstream;
}  // namespace compiled
namespace s11n {
class Stream;
struct SubmodelDeserializeCtx;
}  // namespace s11n
namespace moe {

enum class BehaviorRole {
    EXPERTS,
    DOWNSTREAM,
};

using CompiledExpertsState = std::shared_ptr<ov::npuw::compiled::MoEExperts>;
using CompiledDownstreamState = std::shared_ptr<ov::npuw::compiled::MoEDownstream>;

void put_compiled_experts(v1::subgraphs::Context& context, CompiledExpertsState state);
void put_compiled_downstream(v1::subgraphs::Context& context, CompiledDownstreamState state);

ov::npuw::compiled::MoEExperts* get_compiled_experts(v1::subgraphs::Context& context);
const ov::npuw::compiled::MoEExperts* get_compiled_experts(const v1::subgraphs::Context& context);

ov::npuw::compiled::MoEDownstream* get_compiled_downstream(v1::subgraphs::Context& context);
const ov::npuw::compiled::MoEDownstream* get_compiled_downstream(const v1::subgraphs::Context& context);

bool has_compiled_experts(const v1::subgraphs::CompiledPipeline& pipeline);
bool has_compiled_downstream(const v1::subgraphs::CompiledPipeline& pipeline);
bool has_compiled_state(const v1::subgraphs::CompiledPipeline& pipeline);

void serialize_compiled_state(v1::subgraphs::Context& context,
                              ov::npuw::s11n::Stream& stream,
                              const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx);

std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> register_patterns(
    ov::npuw::v1::subgraphs::PatternRegistry& registry,
    std::size_t moe_chunk_size);

}  // namespace moe
}  // namespace npuw
}  // namespace ov
