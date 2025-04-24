#pragma once

// #include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/iremote_context.hpp"
// #include "remote_tensor.hpp"

#include <string>
#include <map>
#include <memory>
#include <atomic>

namespace ov {
namespace hetero {
class HeteroContext : public ov::IRemoteContext {
public:
    using Ptr = std::shared_ptr<HeteroContext>;

    HeteroContext(std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts);

    // ov::SoPtr<ov::IRemoteTensor> create_tensor(
    //     const ov::element::Type& type,
    //     const ov::Shape& shape,
    //     const ov::AnyMap& params) override {
    //     std::cout << "create tensor with hetero\n";
    //     throw std::runtime_error("HeteroContext does not support create_tensor");
    // }

    const std::string& get_device_name() const override {
        static const std::string name = "HETERO";
        return name;
    }
    const ov::AnyMap& get_property() const override;

    // ov::SoPtr<ov::ITensor> create_host_tensor(const ov::element::Type type, const ov::Shape& shape) override;
    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params) override;

private:
    std::shared_ptr<HeteroContext> get_this_shared_ptr();
    std::map<std::string, ov::SoPtr<ov::IRemoteContext>> m_contexts;
};

}  // namespace intel_gpu
}  // namespace ov