// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/pass/constant_folding.hpp"

namespace ov {
class Model;
class Node;
}  // namespace ov

namespace ov::util {

/**
 * @brief Weights-prefetch observer for the ConstantFolding pass.
 *
 * Loads the mmap-backed weights consumed by a foldable node shortly before the node is folded and
 * evicts them right after, keeping the amount of resident prefetched data under a byte budget. The
 * folding of the tracked nodes is deferred to the ordered folding loop so the data is touched in a
 * predictable order that can overlap I/O with compute.
 *
 * @code
 * auto prefetch = std::make_shared<ov::util::ConstantFoldingWeightsPrefetch>(budget);
 * manager.get_pass_config()->set_pass_extension<ov::pass::ConstantFolding,
 *                                               ov::pass::ConstantFolding::Observer>(prefetch);
 * manager.run_passes(model);
 * prefetch->finish();
 * @endcode
 *
 * Not thread safe.
 */
class OPENVINO_API ConstantFoldingWeightsPrefetch : public ov::pass::ConstantFolding::Observer {
public:
    /**
     * @param byte_budget Upper bound on the bytes of prefetched-but-not-yet-consumed weights. The
     *                    budget may be exceeded only to keep a single oversized constant in flight.
     *                    Use 0 to disable prefetching.
     */
    explicit ConstantFoldingWeightsPrefetch(size_t byte_budget);
    ~ConstantFoldingWeightsPrefetch() override;

    ConstantFoldingWeightsPrefetch(const ConstantFoldingWeightsPrefetch&) = delete;
    ConstantFoldingWeightsPrefetch& operator=(const ConstantFoldingWeightsPrefetch&) = delete;

    void on_pass_begin(const std::shared_ptr<ov::Model>& model) noexcept override;
    bool defer_pre_calculated_values(const std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<const ov::Node>& node) noexcept override;
    void before_ordered_node(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::Node>& node) noexcept override;
    void on_pass_end(const std::shared_ptr<ov::Model>& model) noexcept override;

    void finish() noexcept;

private:
    class Plan;

    size_t m_byte_budget;
    std::vector<std::unique_ptr<Plan>> m_plans;
};

}  // namespace ov::util
