// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNode : public MKLDNNNode {
public:
    MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNGatherNode() override = default;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    struct f32toUi32 {
        inline unsigned int operator()(const float value) {
            return static_cast<unsigned int>(value);
        }
    };

    struct i32toUi32 {
        inline unsigned int operator()(const int32_t value) {
            return static_cast<unsigned int>(value);
        }
    };

    int axis = 0;
    size_t numDictionaries = 1;
    size_t indexRange = 0;
    size_t dataLength = 1;
    static const size_t GATHER_DICTIONARY = 0;
    static const size_t GATHER_INDEXES = 1;
    static const size_t GATHER_AXIS = 2;

    std::string errorPrefix_;

    template <typename index_t, class Conversion>
    void gather();
};

}  // namespace MKLDNNPlugin
