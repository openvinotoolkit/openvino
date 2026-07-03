#pragma once

#include <map>
#include <string>

#include "intel_npu/common/igraph.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class IBlobFormatHandler {
public:
    explicit IBlobFormatHandler(const ov::Tensor& npu_formatted_blob);

    virtual std::shared_ptr<ov::Model> create_dummy_model() = 0;

    virtual std::shared_ptr<IGraph> create_graph() = 0;

    virtual ~IBlobFormatHandler() = default;

private:
    virtual ov::Tensor extract_main_schedule() = 0;

    virtual ov::Tensor extract_init_schedules() = 0;

    virtual ov::Tensor decrypt_schedules() = 0;

    virtual ov::Tensor create_weights_map() = 0;

    ov::Tensor m_npu_formatted_blob;
};

// class RawBlobFormatHandler : public IBlobFormatHandler {
// }

// class BlobFormatV1Handler : public IBlobFormatHandler {
// }

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream npu_formatted_blob);

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob);

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
