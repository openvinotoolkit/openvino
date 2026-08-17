// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_utils/gpu_local_memory.hpp"

#include <gtest/gtest.h>

#include <sstream>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace tests;

TEST(gpu_local_memory, maps_logical_bytes_to_pipeline_specialization) {
    gpu_local_memory_contract contract;
    contract.add({256, 4, gpu_local_memory_mapping::specialization_constant, 7, true});

    kernel_arguments_desc descriptor;
    local_memory_args_desc backend_arguments;
    contract.materialize(descriptor, backend_arguments, 32 * 1024);
    contract.materialize(descriptor, backend_arguments, 32 * 1024);

    ASSERT_EQ(descriptor.specialization_constants.size(), 1u);
    EXPECT_EQ(descriptor.specialization_constants[0].id, 7u);
    EXPECT_EQ(descriptor.specialization_constants[0].value, 64u);
    EXPECT_TRUE(backend_arguments.empty());
}

TEST(gpu_local_memory, maps_logical_bytes_to_existing_backend_argument_contract) {
    gpu_local_memory_contract contract;
    contract.add({768, 1, gpu_local_memory_mapping::backend_argument, 2, true});

    kernel_arguments_desc descriptor;
    local_memory_args_desc backend_arguments;
    contract.materialize(descriptor, backend_arguments, 32 * 1024);

    ASSERT_EQ(backend_arguments.size(), 3u);
    EXPECT_EQ(backend_arguments[2], 768u);
    ASSERT_EQ(descriptor.arguments.size(), 1u);
    EXPECT_EQ(descriptor.arguments[0].t, argument_desc::Types::LOCAL_MEMORY_SIZE);
    EXPECT_EQ(descriptor.arguments[0].index, 2u);
}

TEST(gpu_local_memory, rejects_runtime_size_for_static_shader) {
    gpu_local_memory_contract contract;
    contract.add({256, 4, gpu_local_memory_mapping::static_shader, 0, true});

    kernel_arguments_desc descriptor;
    local_memory_args_desc backend_arguments;
    EXPECT_THROW(contract.materialize(descriptor, backend_arguments, 32 * 1024), ov::Exception);
}

TEST(gpu_local_memory, serialization_round_trip_preserves_pipeline_mapping) {
    gpu_local_memory_contract original;
    original.add({1024, 4, gpu_local_memory_mapping::specialization_constant, 3, true});
    original.add({512, 1, gpu_local_memory_mapping::backend_argument, 1, false});

    std::stringstream storage(std::ios::in | std::ios::out | std::ios::binary);
    BinaryOutputBuffer output(storage);
    original.save(output);
    storage.seekg(0);

    gpu_local_memory_contract restored;
    BinaryInputBuffer input(storage, get_test_engine());
    restored.load(input);

    ASSERT_EQ(restored.size(), 2u);
    EXPECT_EQ(restored[0].byte_size, 1024u);
    EXPECT_EQ(restored[0].element_size, 4u);
    EXPECT_EQ(restored[0].mapping, gpu_local_memory_mapping::specialization_constant);
    EXPECT_EQ(restored[0].mapping_id, 3u);
    EXPECT_TRUE(restored[0].runtime_resolved);
    EXPECT_EQ(restored[1].mapping, gpu_local_memory_mapping::backend_argument);
    EXPECT_EQ(restored[1].mapping_id, 1u);
}
