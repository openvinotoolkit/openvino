// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// #include "openvino/op/op.hpp"
// #include "openvino/core/cache_manager.hpp"

// namespace ov {
// namespace op {

// /// \brief PagedCacheDistributor operation distributes cache inputs to the cache-managed node.
// ///
// /// \ingroup ov_ops_cpp_api
// ///
// /// This operation computes attention using a paged memory model, allowing efficient handling of long sequences.
// class OPENVINO_API PagedCacheDistributor : public ov::op::Op {
// public:
//     OPENVINO_OP("PagedCacheDistributor");

//     PagedCacheDistributor() = default;

//     /// \brief Constructs a PagedCacheDistributor operation.
//     ///
//     /// \param args Input arguments vector containing:
//     ///             - key_cache
//     ///             - value_cache
//     ///             - past_lens
//     ///             - max_context_len
//     PagedCacheDistributor(const ov::OutputVector& args);


//     void validate_and_infer_types() override;
//     std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

//     /// \brief Gets the output element type at the specified index.
//     const ov::element::Type get_out_type() const;

//     /// \brief Sets the output element type at the specified index.
//     void set_out_type(int index, const ov::element::Type& output_type);

//     const std::shared_ptr<ov::CacheManager> get_cache_manager() const;

//     void set_cache_manager(std::shared_ptr<ov::CacheManager> cache_manager_ptr);

// protected:
//     // key_cache, value_cache
//     std::vector<ov::element::Type> m_output_type = {ov::element::f32, ov::element::f32};
//     std::shared_ptr<ov::CacheManager> cache_manager = nullptr;
// };

}  // namespace op
}  // namespace ov
