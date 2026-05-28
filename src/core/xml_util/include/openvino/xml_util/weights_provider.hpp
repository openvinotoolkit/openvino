// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <map>
#include <memory>

namespace ov {
class AlignedBuffer;
}  // namespace ov

namespace ov::util {

class WeightsProvider {
public:
    virtual ~WeightsProvider() = default;

    virtual std::shared_ptr<ov::AlignedBuffer> load_region(size_t offset, size_t size) = 0;
    virtual size_t size() const = 0;
};

class BufferWeightsProvider : public WeightsProvider {
public:
    explicit BufferWeightsProvider(std::shared_ptr<ov::AlignedBuffer> weights);

    std::shared_ptr<ov::AlignedBuffer> load_region(size_t offset, size_t size) override;
    size_t size() const override;

private:
    std::shared_ptr<ov::AlignedBuffer> m_weights;
};

class FileWeightsProvider : public WeightsProvider {
public:
    explicit FileWeightsProvider(std::filesystem::path weights_path);

    std::shared_ptr<ov::AlignedBuffer> load_region(size_t offset, size_t size) override;
    size_t size() const override;

private:
    using WeightsRegionKey = std::pair<size_t, size_t>;

    std::filesystem::path m_weights_path;
    size_t m_weights_size = 0;
    size_t m_weights_source_id = 0;
    std::shared_ptr<ov::AlignedBuffer> m_weights_source_handle;
    std::map<WeightsRegionKey, std::shared_ptr<ov::AlignedBuffer>> m_loaded_weights_regions;
};
}  // namespace ov::util
