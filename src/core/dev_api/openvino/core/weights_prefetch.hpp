// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "openvino/core/core_visibility.hpp"

namespace ov {
class Model;
class Node;
namespace op::v0 {
class Constant;
}  // namespace op::v0
}  // namespace ov

namespace ov::weight_sharing {

/**
 * @brief Populates the pages backing the weights of a model, exactly once.
 *
 * Weights mapped from a file are faulted in lazily, one page at a time, by whichever thread happens
 * to touch them first. This class asks the OS to load them up front and in the background instead,
 * so the loading overlaps with the rest of the work rather than serializing inside it.
 *
 * Registration and prefetching are separate steps: a consumer registers the constants as soon as
 * they are known and picks the moment to start the transfer. A single instance may be shared by
 * several graphs built from the same model, so the weights are prefetched only once.
 *
 * All the calls are thread safe.
 */
class OPENVINO_API WeightsPrefetch {
public:
    using Ptr = std::shared_ptr<WeightsPrefetch>;

    /**
     * @brief Registers the constants of @p model which are worth prefetching.
     *
     * Constants read by index at inference time are filtered out, see @ref register_constants(const
     * std::vector<std::shared_ptr<ov::Node>>&).
     *
     * @param model Model to register the weights of.
     */
    void register_constants(const ov::Model& model);

    /**
     * @brief Registers the constants found in @p ops which are worth prefetching.
     *
     * Constants whose every consumer addresses them by index (embedding tables and alike) are
     * filtered out: only a fraction of such a constant is ever read, so populating all of its pages
     * would only inflate the resident memory. Duplicates and constants already registered are
     * ignored, so graphs replicated from the very same model can register concurrently.
     *
     * @param ops Nodes to look for constants in.
     */
    void register_constants(const std::vector<std::shared_ptr<ov::Node>>& ops);

    /**
     * @brief Starts loading the registered weights in the background and releases the registry.
     *
     * Returns immediately, the data is not guaranteed to be resident when the call returns.
     * Subsequent calls are no-ops.
     */
    void prefetch_once();

private:
    // Observers only: a class whose job is to touch weight pages must not keep them alive.
    using ConstantRef = std::weak_ptr<const ov::op::v0::Constant>;

    std::once_flag m_once;
    std::mutex m_mutex;
    std::set<ConstantRef, std::owner_less<ConstantRef>> m_registered;
    std::vector<ConstantRef> m_constants;
};

}  // namespace ov::weight_sharing

namespace ov {
namespace wsh = ov::weight_sharing;
}
