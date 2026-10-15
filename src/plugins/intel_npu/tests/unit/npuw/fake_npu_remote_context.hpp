// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test::npuw {

// Emulates a remote tensor backed by host memory (either user-provided or owned).
class FakeRemoteTensor : public ov::IRemoteTensor {
public:
    FakeRemoteTensor(std::shared_ptr<ov::ITensor> host, std::string device_name);

    using ov::IRemoteTensor::copy_from;
    using ov::IRemoteTensor::copy_to;

    void set_shape(ov::Shape shape) override;
    const ov::element::Type& get_element_type() const override;
    const ov::Shape& get_shape() const override;
    const ov::Strides& get_strides() const override;
    const ov::AnyMap& get_properties() const override;
    const std::string& get_device_name() const override;
    void copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t, size_t, const ov::Shape&) const override;
    void copy_from(const std::shared_ptr<const ov::ITensor>& src, size_t, size_t, const ov::Shape&) override;

    const void* host_data() const;

private:
    std::shared_ptr<ov::ITensor> m_host;
    ov::AnyMap m_properties;
    std::string m_device_name;
};


class FakeNpuRemoteTensor : public FakeRemoteTensor {
public:
    explicit FakeNpuRemoteTensor(std::shared_ptr<ov::ITensor> host);
};

struct Statistics {
    std::atomic<size_t> allocated_remote_tensors{0};
    std::atomic<size_t> imported_remote_tensors{0};
};

struct StatisticsSnapshot {
    StatisticsSnapshot(const Statistics& stats) :
            allocated_remote_tensors(stats.allocated_remote_tensors.load()),
            imported_remote_tensors(stats.imported_remote_tensors.load()) {}
    size_t allocated_remote_tensors{0};
    size_t imported_remote_tensors{0};
};

// Wraps ov::intel_npu::mem_handle memory when provided, otherwise allocates host memory.
class FakeRemoteContext : public ov::IRemoteContext {
public:
    explicit FakeRemoteContext(std::string device_name);

    const std::string& get_device_name() const override;
    const ov::AnyMap& get_property() const override;

    StatisticsSnapshot get_statistics_snapshot() const { return m_statistics; }
protected:
    virtual std::shared_ptr<FakeRemoteTensor> wrap_host_tensor(std::shared_ptr<ov::ITensor> host) const;
    void increment_allocated_remote_tensors() { m_statistics.allocated_remote_tensors++; }
    void increment_imported_remote_tensors() { m_statistics.imported_remote_tensors++; }
private:
    std::string m_device_name;
    ov::AnyMap m_properties;
    Statistics m_statistics;
};

class FakeNpuRemoteContext : public FakeRemoteContext {
public:
    FakeNpuRemoteContext();
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params) override;

protected:
    std::shared_ptr<FakeRemoteTensor> wrap_host_tensor(std::shared_ptr<ov::ITensor> host) const override;
};

// Returns the underlying host pointer for both regular and FakeRemoteTensor-backed tensors.
const void* tensor_data_ptr(const ov::Tensor& tensor);

}  // namespace ov::test::npuw
