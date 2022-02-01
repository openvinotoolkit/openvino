// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>
#include <list>
#include <vector>

#include <vpu/utils/enums.hpp>
#include <vpu/model/stage.hpp>
#include <vpu/model/data.hpp>
#include <vpu/model/edges.hpp>
#include <vpu/middleend/allocator/structs.hpp>

namespace vpu {

//
// AllocatorForShaves
//

class AllocatorForShaves final {
public:
    explicit AllocatorForShaves(allocator::MemoryPool &cmxMemoryPool);

    void reset();

    bool allocateSHAVEs(
                const Stage& stage,
                StageSHAVEsRequirements reqs);
    void freeSHAVEs();

    int getLockedSHAVEs() const { return _lockedSHAVEs; }

    void selfCheck();

private:
    int _lockedSHAVEs = 0;

    allocator::MemoryPool &_cmxMemoryPool;
};

}  // namespace vpu
