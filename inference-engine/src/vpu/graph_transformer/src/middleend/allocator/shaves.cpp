// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/allocator/shaves.hpp>

#include <unordered_set>
#include <algorithm>
#include <limits>
#include <set>

#include <vpu/compile_env.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

namespace {

// TODO: investigate the value
    const int SHAVES_LIMITATION_FOR_HW = 2;

}  // namespace

AllocatorForShaves::AllocatorForShaves(allocator::MemoryPool &cmxMemoryPool): _cmxMemoryPool(cmxMemoryPool) {
}

bool AllocatorForShaves::allocateSHAVEs(
        const Stage& stage,
        StageSHAVEsRequirements reqs) {
    const auto& env = CompileEnv::get();

    //
    // Check that we don't allocate twice
    //

    if (_lockedSHAVEs != 0) {
        VPU_THROW_EXCEPTION << "Can't allocate SHAVEs : was already allocated";
    }

    //
    // Check stage requirements
    //

    if (reqs == StageSHAVEsRequirements::NotNeeded) {
        // Stage doesn't need SHAVEs.
        return true;
    }

    //
    // Check the amount of free SHAVEs
    //

    auto usedCMXslices = (_cmxMemoryPool.curMemOffset + CMX_SLICE_SIZE - 1) / CMX_SLICE_SIZE;
    IE_ASSERT(usedCMXslices <= env.resources.numCMXSlices);

    const auto numAvailableSHAVEs = std::min(env.resources.numCMXSlices - usedCMXslices, env.resources.numSHAVEs);
    if (numAvailableSHAVEs == 0) {
        return false;
    }

    int necessarySHAVEsNum = numAvailableSHAVEs;
    if (reqs == StageSHAVEsRequirements::NeedMax) {
        if (numAvailableSHAVEs < env.resources.numSHAVEs) {
            return false;
        }
    } else if (reqs == StageSHAVEsRequirements::OnlyOne) {
        necessarySHAVEsNum = 1;
    } else if (reqs == StageSHAVEsRequirements::TwoOrOne) {
        necessarySHAVEsNum = std::min(numAvailableSHAVEs, 2);
    } else if (reqs == StageSHAVEsRequirements::CanBeLimited) {
        bool needToLimit = false;
        if (stage->category() == StageCategory::HW) {
            needToLimit = true;
        }
        for (const auto& prevStage : stage->prevStages()) {
            if (prevStage->category() == StageCategory::HW) {
                needToLimit = true;
                break;
            }
        }
        for (const auto& nextStage : stage->nextStages()) {
            if (nextStage->category() == StageCategory::HW) {
                needToLimit = true;
                break;
            }
        }

        if (needToLimit) {
            necessarySHAVEsNum = std::min(numAvailableSHAVEs, SHAVES_LIMITATION_FOR_HW);
        }
    }

    //
    // Lock SHAVEs
    //

    _lockedSHAVEs = necessarySHAVEsNum;

    stage->setNumSHAVEs(_lockedSHAVEs);

    return true;
}

void AllocatorForShaves::freeSHAVEs() {
    _lockedSHAVEs = 0;
}

void AllocatorForShaves::reset() {
    _lockedSHAVEs = 0;
}

void AllocatorForShaves::selfCheck() {
    if (_lockedSHAVEs > 0) {
        VPU_THROW_EXCEPTION << "Internal error in SHAVEs allocation";
    }
}
}  // namespace vpu
