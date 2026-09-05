// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

struct PushConstants {
    float increment;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> input_data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output_data;
[[vk::push_constant]] ConstantBuffer<PushConstants> push_constants;

groupshared float local_values[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void slang_bootstrap(uint3 global_id : SV_DispatchThreadID, uint local_index : SV_GroupThreadID) {
    local_values[local_index] = input_data[global_id.x];
    GroupMemoryBarrierWithGroupSync();
    const uint adjacent_index = (local_index + 1) % 64;
    output_data[global_id.x] = local_values[adjacent_index] + push_constants.increment;
}
