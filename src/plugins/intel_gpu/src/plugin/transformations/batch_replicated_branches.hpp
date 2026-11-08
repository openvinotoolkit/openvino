// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Runs N equal branches that process slices of one tensor as a single branch with batch N.
 *
 * Multi-camera models (e.g. BEV detectors) run the same backbone on every camera image and concatenate the
 * results. Each replica is a separate batch-1 Convolution, which underuses the GPU; one batch-N Convolution
 * computes the same values in one kernel.
 *
 * Before ([ ] marks optional nodes):
 *
 *                       data [.., N, ..]
 *                │                         │
 *          ┌─────┴─────┐             ┌─────┴─────┐
 *          │ Gather(0) │             │Gather(N-1)│
 *          └─────┬─────┘             └─────┬─────┘
 *         ┌──────┴──────┐           ┌──────┴──────┐
 *         │ Convolution │    ...    │ Convolution │
 *         └──────┬──────┘           └──────┬──────┘
 *            [MaxPool]                 [MaxPool]
 *              [Add]                     [Add]
 *             [Relu]                    [Relu]
 *                │ [1,C,H,W]               │ [1,C,H,W]
 *                └────────────┬────────────┘
 *                   ┌─────────┴─────────┐
 *                   │ Concat(axis k > 0)│
 *                   └───────────────────┘
 *
 * After:
 *
 *                       data [.., N, ..]
 *                             │
 *                        [Transpose]       replica axis to the front
 *                             │
 *                       ┌─────┴─────┐
 *                       │  Reshape  │      [N,C,H,W]
 *                       └─────┬─────┘
 *                      ┌──────┴──────┐
 *                      │ Convolution │     weights of branch 0
 *                      └──────┬──────┘
 *                         [MaxPool]
 *                           [Add]
 *                          [Relu]
 *                       ┌─────┴─────┐
 *                       │ Transpose │      N in front of axis k
 *                       └─────┬─────┘
 *                       ┌─────┴─────┐
 *                       │  Reshape  │      N merged into axis k
 *                       └───────────┘
 *
 * The rewrite is applied when:
 *  - all shapes are static and every branch output is [1, C, H, W] with Concat on axis k > 0, so batch N is
 *    free for the replicas and the Concat layout can be restored by Transpose and Reshape;
 *  - branch i gathers slice i of one tensor and the N slices cover the gathered dimension, so the batched
 *    input is that tensor with the replica axis moved to the front; Transpose is added only when a dim before
 *    that axis is not 1;
 *  - all branches have the same operations, attributes and constants, so the batched branch computes every
 *    replica; Add uses numpy broadcast;
 *  - every branch node feeds only the next one and MaxPool indices are unused, otherwise the original branches
 *    would stay in the graph next to the batched one;
 *  - branch nodes and Concat are not TypeRelaxed: Gather and Concat are replaced by new Reshape and Transpose
 *    nodes, which would drop a type override;
 *  - branch nodes have no control dependencies or dependents: cloning does not keep the dependencies, and the
 *    dependents would keep the original branch alive.
 */
class BatchReplicatedBranches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::BatchReplicatedBranches");
    BatchReplicatedBranches();
};

}  // namespace ov::intel_gpu
