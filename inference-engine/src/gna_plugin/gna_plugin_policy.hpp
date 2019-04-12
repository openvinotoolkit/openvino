// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


namespace GNAPluginNS {
/**
 * @brief policy agregates various settings that cannot be tweak using configuration options right now,
 * and essential to keep test coverage for options both in on and off cases
 */
class Policy {
 public:
    /**
    * @brief for scaleshift substitution, weight tiling simplify final graph but have extra weights overhead
    * if not defined scaleshift broadcast will result in creating multiple diagonal layers instead of weight tiling
    */
    enum {
        WEIGHTS_TILING,
        /**
         * GNA has limited amount of batch so even existed topologies cannot be substituted with only batching,
         * this option combines batch and weights tiling
         */
        BATCH_AND_WEIGHTS_TILING,
        DIAGLAYER_TILING
    } ScaleShiftPolicy = WEIGHTS_TILING;

    /**
     * Policy on whether to substitute permute layers or not
     */
    enum {
        DISABLED,
        AUTO_PERMUTE
    } PermutePolicy = DISABLED;
};

}  // namespace GNAPluginNS
