// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <variant>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

typedef std::variant<std::shared_ptr<ov::StringAlignedBuffer>, std::shared_ptr<ov::SharedStringAlignedBuffer>, std::shared_ptr<ov::AlignedBuffer>> WeightsVariant;

/**
 * @brief Serialize transformation converts ov::Model into IR files
 * @attention
 * - dynamic shapes are not supported
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API LightSerialize : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("LightSerialize");

    enum class Version : uint8_t {
        UNSPECIFIED = 0,  // Use the latest or function version
        IR_V10 = 10,      // v10 IR
        IR_V11 = 11       // v11 IR
    };
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    LightSerialize(std::ostream& xmlFile,
                   std::map<int64_t, WeightsVariant>& offsetConstMap,
                   Version version = Version::UNSPECIFIED);

private:
    std::ostream* m_xmlFile;
    std::map<int64_t, WeightsVariant>& m_offsetConstMap;
    const Version m_version;
    const std::map<std::string, ov::OpSet> m_custom_opsets;
};

}  // namespace pass
}  // namespace ov
