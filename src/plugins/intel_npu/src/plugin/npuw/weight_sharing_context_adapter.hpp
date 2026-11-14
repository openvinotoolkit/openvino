// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace orc {
class Stream;
}
namespace weights {

ov::Tensor tensor_factory(const ov::ICore& core,
                          const std::string& device,
                          const ov::element::Type& type,
                          const ov::Shape& shape,
                          std::shared_ptr<ov::AlignedBuffer> staging_buffer_ptr = {});

class TransformationWeightContextIDMapper {
public:
    using UID_Type = int64_t;
    using SourceID_Type = ov::weight_sharing::DataID;
    using ConstantID_Type = ov::weight_sharing::DataID;
    using WSContextIDs = std::pair<SourceID_Type, ConstantID_Type>;

    std::optional<WSContextIDs> get_ov_context_ids(UID_Type uid) const;
    std::tuple<bool, WSContextIDs> get_or_create_ov_context_ids(UID_Type uid);
    UID_Type get_transformation_uid(const WSContextIDs& context_ids) const;
    WSContextIDs set_ov_context_ids(UID_Type uid);
    UID_Type set_transformation_uid(const WSContextIDs& context_ids);

private:
    static WSContextIDs create_ov_context_ids(UID_Type uid);
    void insert_mapping(UID_Type uid, const WSContextIDs& context_ids);

    struct ContextIDsHash {
        size_t operator()(const WSContextIDs& context_ids) const {
            return std::hash<SourceID_Type>{}(context_ids.first) ^
                   (std::hash<ConstantID_Type>{}(context_ids.second) << 1);
        }
    };

    mutable std::atomic_flag m_context_ids_lock = ATOMIC_FLAG_INIT;
    std::unordered_map<UID_Type, WSContextIDs> m_ov_context_ids;
    std::unordered_map<WSContextIDs, UID_Type, ContextIDsHash> m_transformation_uids;
};

class WeightSharingContextAccessor {
public:
    using UID_Type = TransformationWeightContextIDMapper::UID_Type;

    WeightSharingContextAccessor(const std::shared_ptr<ov::weight_sharing::Context>& context) : m_context(context) {}

    struct Handle {
        friend WeightSharingContextAccessor;
        ~Handle() = default;
        Handle(Handle&&) noexcept = default;
        Handle& operator=(Handle&&) noexcept = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

        std::shared_ptr<ov::weight_sharing::Context> context;

    private:
        Handle(std::mutex& m, std::shared_ptr<ov::weight_sharing::Context> ctx) : context(ctx), lock(m) {}
        std::unique_lock<std::mutex> lock;
    };

    Handle get_handle() {
        return Handle(m_mutex, m_context);
    }

    bool has_transformation_buffer(UID_Type transformation_uid, Handle& handle);
    ov::weight_sharing::WeightBuffer request_shared_weight_source_buffer(UID_Type transformation_uid,
                                                                         const element::Type& type,
                                                                         const ov::Shape& shape,
                                                                         const std::string& device,
                                                                         Handle& handle);

    /* TODO CVS-199039, CVS-199076
     * Extracts the tensor data and transforms it into a weight source buffer to register it as a shared weight source.
     * The better control flow would involve creating the shared buffer first,
     * which would serve as the staging buffer for ov::Tensor and then registering it.
     * OV API currently doesn't allow to operate in that way.
     *
     * Once the issues are addressed, the workflow can be updated to create the shared buffer first and then register it
     * and the method register_tensor_as_shared_weight_source can be removed.
     */
    ov::weight_sharing::WeightBuffer register_tensor_as_shared_weight_source(UID_Type transformation_uid,
                                                                             const std::string& device,
                                                                             const ov::Tensor& tensor,
                                                                             Handle& handle);

    bool serialize_native_format(ov::npuw::orc::Stream& stream);
    std::set<UID_Type> consume_from(ov::npuw::orc::Stream& stream, const ov::ICore& core);

protected:
    static ov::Tensor tensor_from_shared_buffer(const ov::weight_sharing::WeightBuffer& shared_source);

    /* CVS-198039
     * See a comment below
     */
    const ov::op::v0::Constant& get_transformed_constant(const std::string& device,
                                                         UID_Type transformation_uid,
                                                         Handle& handle) const;
    std::set<UID_Type> consume_from_impl(ov::npuw::orc::Stream& stream, const ov::ICore& core, Handle& handle);

    TransformationWeightContextIDMapper ids_mapper;

private:
    static ov::weight_sharing::WeightBuffer create_shared_buffer(ov::weight_sharing::DataID source_id,
                                                                 std::size_t size);
    static ov::weight_sharing::WeightBuffer create_shared_buffer(ov::weight_sharing::DataID source_id,
                                                                 const ov::Tensor& tensor);

    void register_shared_weight_source_for_weight(ov::weight_sharing::WeightBuffer source_buffer,
                                                  const std::string& device,
                                                  Handle& handle);
    void register_transformed_constant(std::shared_ptr<ov::AlignedBuffer> transformation_buffer,
                                       UID_Type transformation_uid,
                                       const std::string& device,
                                       const element::Type& type,
                                       const ov::Shape& shape,
                                       Handle& handle);
    bool serialize_native_format_impl(ov::npuw::orc::Stream& stream, Handle& handle) const;
    std::mutex m_mutex;
    std::shared_ptr<ov::weight_sharing::Context> m_context = nullptr;
    std::unordered_map<ov::weight_sharing::DataID, ov::weight_sharing::WeightBuffer> m_shared_sources;

    /* CVS-198039
     * WeightMetaData has no m_shape,
     * That is substantial for proper data recovery during deserialization,
     * because the shape information is necessary to correctly reconstruct the tensor.
     *
     * As a WA we store the shape information separately to ensure proper reconstruction of the tensor during
     * deserialization. once the issue is addressed, the TransformedWeightsMap containing the shape information can be
     * removed.
     */
    using TransformedWeightsMap = std::unordered_map<UID_Type, ov::op::v0::Constant>;
    std::unordered_map<std::string /* device */, TransformedWeightsMap> m_transformed_weights_per_device;
};

}  // namespace weights
}  // namespace npuw
}  // namespace ov
