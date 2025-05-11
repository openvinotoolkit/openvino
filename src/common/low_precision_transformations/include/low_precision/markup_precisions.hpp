// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "low_precision/lpt_visibility.hpp"
#include "low_precision/common/precisions_restriction.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupPrecisions;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

// Transformation is used to add customization options runtime
/**
 * @ingroup ov_transformation_common_api
 * @brief MarkupPrecisions transformation marks:
 *    1) not supported operations by PrecisionsAttribute attribute with empty precisions,
 *    2) operations with required precisions by PrecisionsAttribute attribute according to the provided restrictions,
 *    3) precision preserved operations by PrecisionPreservedAttribute attribute.
 *
 * For more details about the transformation, refer to
 * [MarkupPrecisions](@ref openvino_docs_OV_UG_lpt_MarkupPrecisions) page
 * in the OpenVINO Developer Guide.
 */
class ov::pass::low_precision::MarkupPrecisions : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("low_precision::MarkupPrecisions");
    class Restriction {
    public:
        class RestrictionByVersion {
        public:
            RestrictionByVersion() = default;
            RestrictionByVersion(
                const std::function<PrecisionsRestriction::PrecisionsByPorts(const std::shared_ptr<Node>&)>& precisionsFunction,
                const PrecisionsRestriction::PrecisionsByPorts& precisions) :
                precisionsFunction(precisionsFunction),
                precisions(precisions) {}

            PrecisionsRestriction::PrecisionsByPorts get(const std::shared_ptr<Node>& node) const {
                return (precisionsFunction != nullptr) ? precisionsFunction(node) : precisions;
            }

        private:
            std::function<PrecisionsRestriction::PrecisionsByPorts(const std::shared_ptr<Node>&)> precisionsFunction;
            PrecisionsRestriction::PrecisionsByPorts precisions;
        };

        explicit Restriction(const bool versionIsRequired) : versionIsRequired(versionIsRequired) {}
        void add(const std::string version_id, const RestrictionByVersion& precisions) {
            precisionsByVersion.emplace(version_id, precisions);
        }

        bool versionIsRequired;
        std::unordered_map<std::string, RestrictionByVersion> precisionsByVersion;
    };

    explicit MarkupPrecisions(const std::vector<PrecisionsRestriction>& restrictions = {},
        const std::vector<ov::element::Type>& defaultPrecisions = { ov::element::u8, ov::element::i8 });
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    static bool isPrecisionPreserved(const std::shared_ptr<Node>& node);
    static bool isSupported(const std::shared_ptr<Node>& node);
    std::unordered_map<std::string, Restriction> restrictionsByOperation;
    std::vector<ov::element::Type> defaultPrecisions;
};
