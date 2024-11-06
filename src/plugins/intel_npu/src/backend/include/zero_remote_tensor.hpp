// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "intel_npu/common/remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

class ZeroRemoteTensor : public RemoteTensor {
public:
    ZeroRemoteTensor(std::shared_ptr<ov::IRemoteContext> context,
                     std::shared_ptr<ZeroInitStructsHolder> init_structs,
                     const ov::element::Type& element_type,
                     const ov::Shape& shape,
                     const Config& config,
                     ov::intel_npu::TensorType tensor_type = ov::intel_npu::TensorType::BINDED,
                     ov::intel_npu::MemType mem_type = ov::intel_npu::MemType::L0_INTERNAL_BUF,
                     void* mem = nullptr);

    ~ZeroRemoteTensor() override;

private:
    void allocate(const size_t bytes) override;
    bool deallocate() noexcept override;
    bool is_allocated() const noexcept;
    void update_properties();

    const Config _config;
    Logger _logger;

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    ze_device_properties_t _ze_properties = {};

    ov::intel_npu::TensorType _tensor_type;
    ov::intel_npu::MemType _mem_type;
    void* _mem = nullptr;
    void* _data = nullptr;

    bool _external_memory_support = false;
};

}  // namespace intel_npu
