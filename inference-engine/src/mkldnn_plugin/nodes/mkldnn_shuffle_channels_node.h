// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

#define CNTR_SIZE 3

enum ShuffleChannelsLayoutType {
    planar_layout,
    blocked_layout,
    by_channel_layout
};

struct jit_shuffle_channels_config_params {
    bool shuffle_innermost; // shuffle the innermost dimension
    bool permute_mode;      // true: use instructions (permute, shuffle*); false: use instructions gather
    size_t group;
    size_t shuffle_size;    // size of the dimension to shuffle
    size_t channel_batch;   // channel batch only for blocked layout
    size_t shuffle_stride;  // shuffle stride only for blocked layout
    ShuffleChannelsLayoutType layout;
    mkldnn::memory::data_type data_type;
    int data_size;
};

struct jit_shuffle_channels_call_args {
    const void *src;
    const int *index;
    const int *tab_idx;
    void *dst;
    size_t work_amount;
};

struct jit_uni_shuffle_channels_kernel {
    void (*ker_)(const jit_shuffle_channels_call_args *);

    void operator()(const jit_shuffle_channels_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_shuffle_channels_kernel(jit_shuffle_channels_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_shuffle_channels_kernel() {}

    jit_shuffle_channels_config_params jcp_;
};

class MKLDNNShuffleChannelsNode : public MKLDNNNode {
public:
    MKLDNNShuffleChannelsNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNShuffleChannelsNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    void shuffle_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void shuffle_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void shuffle_ref(const float *in_ptr, float *out_ptr);
    inline void shuffle_kernel(const uint8_t *in_p, uint8_t *out_p, const int *src_idx, const int *tab_idx, size_t work_amount);
    inline size_t initter(size_t start, size_t size, size_t *counters, size_t *own_dims, size_t *ownStrides);
    inline size_t updater(size_t idx, size_t size, size_t *counters, size_t *own_dims, size_t *ownStrides);

    bool shuffle_innermost = false; // shuffle the innermost dimension
    bool permute_mode = false;       // true: use instructions (permute, shuffle*); false: use instructions gather
    bool jit_mode = true;
    int axis;
    size_t group;
    size_t group_size;   // size of each group
    size_t shuffle_size; // size of the dimension to shuffle
    size_t blk_size;
    size_t dims_size;
    size_t data_size;
    size_t dataLength = 1;
    size_t work_amount_dst;
    size_t own_dims[CNTR_SIZE];
    size_t ownStrides[CNTR_SIZE];
    InferenceEngine::SizeVector dst_dims;
    ShuffleChannelsLayoutType layout;
    mkldnn::memory::data_type data_type;

    std::shared_ptr<jit_uni_shuffle_channels_kernel> shuffle_channels_kernel;
};

}  // namespace MKLDNNPlugin
