// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ConvertPagedAttnInputs;

/**
 * @ingroup ov_transformation_common_api
 * @brief Set precision and shape of KV cache in PagedAttn op based runtime options
 */

class ConvertPagedAttnInputs : public ov::pass::MatcherPass {
public:
    using UpdateShapeFunc = std::function<void(const ov::element::Type, const bool, const size_t, int64_t&, int64_t&)>;

    struct KVCacheConfig {
        ov::element::Type keyCachePrecision;
        ov::element::Type valueCachePrecision;
        ov::element::Type inferencePrecision;
        size_t keyCacheBlockSize = 32;
        size_t valueCacheBlockSize = 32;
        size_t keyCacheGroupSize = 0;
        size_t valueCacheGroupSize = 0;
        bool keyCacheQuantBychannel = false;
        bool valueCacheQuantBychannel = false;
        std::vector<size_t> keyCacheDimOrder = {0, 1, 2, 3};
        std::vector<size_t> valueCacheDimOrder = {0, 1, 2, 3};
    };

    OPENVINO_MATCHER_PASS_RTTI("ConvertPagedAttnInputs");
    ConvertPagedAttnInputs(const KVCacheConfig& config, UpdateShapeFunc update_shape_func);

    void setKVCacheConfig(const KVCacheConfig& config);

    const KVCacheConfig& getKVCacheConfig() const;

private:
    KVCacheConfig m_config;
    UpdateShapeFunc m_update_shape_func;
};

}  // namespace pass
}  // namespace ov
