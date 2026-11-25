// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
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

ov::Tensor tensor_factory(std::shared_ptr<ov::IRemoteContext> remote_ctx,
                          const ov::element::Type& type,
                          const ov::Shape& shape,
                          std::shared_ptr<ov::AlignedBuffer> staging_buffer_ptr = {});

// Maps NPUW transformation UIDs to the OpenVINO weight-sharing context IDs that are represented by the pair of source
// and constant IDs. The direct mapping is needed to associate transformation UIDs with their corresponding sources of
// constants in the weight-sharing context during the evaluation procedure. The reverse mapping is needed to serialize
// to a NPUW blob and deserialize from a NPUw blob using the weight-sharing context. This class is not thread-safe.
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

    std::unordered_map<UID_Type, WSContextIDs> m_ov_context_ids;
    std::unordered_map<WSContextIDs, UID_Type, ContextIDsHash> m_transformation_uids;
};

// Adapts an OpenVINO weight-sharing context for NPUW transformed weights.
// It must be called during the evaluations of LazyTensors, to register transformed weights as shared weight sources
// in ov::weight_sharing::Context, retains metadata needed to reconstruct transformed tensors, and serializes or
// restores weights in NPUW blobs. This class does not provide synchronization; the caller ov::npuw::weights::Bank must
// serialize concurrent access itself.
class WeightSharingContextAccessor {
public:
    using UID_Type = TransformationWeightContextIDMapper::UID_Type;

    WeightSharingContextAccessor(const std::shared_ptr<ov::weight_sharing::Context>& context) : m_context(context) {}

    bool has_transformation_buffer(UID_Type transformation_uid);
    ov::weight_sharing::WeightBuffer request_shared_weight_source_buffer(UID_Type transformation_uid,
                                                                         const element::Type& type,
                                                                         const ov::Shape& shape,
                                                                         const std::string& device);

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
                                                                             const ov::Tensor& tensor);

    bool serialize_native_format(ov::npuw::orc::Stream& stream);
    std::set<UID_Type> consume_from(ov::npuw::orc::Stream& stream, const ov::ICore& core);

protected:
    static ov::Tensor tensor_from_shared_buffer(const ov::weight_sharing::WeightBuffer& shared_source);

    /* CVS-198039
     * See a comment below
     */
    const ov::op::v0::Constant& get_transformed_constant(const std::string& device, UID_Type transformation_uid) const;

    TransformationWeightContextIDMapper ids_mapper;

private:
    friend class Bank;

    static ov::weight_sharing::WeightBuffer create_shared_buffer(ov::weight_sharing::DataID source_id,
                                                                 std::size_t size);
    static ov::weight_sharing::WeightBuffer create_shared_buffer(ov::weight_sharing::DataID source_id,
                                                                 const ov::Tensor& tensor);

    void register_shared_weight_source_for_weight(ov::weight_sharing::WeightBuffer source_buffer,
                                                  const std::string& device);
    void register_transformed_constant(std::shared_ptr<ov::AlignedBuffer> transformation_buffer,
                                       UID_Type transformation_uid,
                                       const std::string& device,
                                       const element::Type& type,
                                       const ov::Shape& shape);
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
