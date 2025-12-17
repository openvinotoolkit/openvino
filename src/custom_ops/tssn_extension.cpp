#include "openvino/core/extension.hpp"
#include "openvino/core/op_extension.hpp"
#include "tssn_op.hpp"
#include "composite_tssn.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<ov::op::v0::TSSN>>(),
        std::make_shared<ov::OpExtension<ov::op::v0::CompositeTSSN>>()
    })
);
