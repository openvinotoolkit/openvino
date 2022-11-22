// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util.hpp"

using namespace GNAPluginNS::GNALimitations;
using namespace ov::intel_gna;

namespace {
std::ostream& operator<<(std::ostream& os, const std::set<ov::element::Type>& t) {
    for (auto it = t.begin(); it != t.end(); ++it) {
        if (it != t.begin()) {
            os << ", " << *it;
        } else {
            os << *it;
        }
    }
    return os;
}

bool is_precision_supported(const std::shared_ptr<ov::Node>& node,
                            const std::set<ov::element::Type>& supported_types,
                            bool is_exception_allowed) {
    if (!node) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "Node is empty!\n";
        }
        return false;
    }
    if (supported_types.count(node->get_element_type()) == 0) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "The plugin does not support input precision with " +
                std::string(node->get_type_name()) + " format. Supported input precisions " << supported_types << "\n";
        }
        return false;
    }
    return true;
}

bool is_conv_supported(const std::shared_ptr<ngraph::op::ConvolutionIE>& conv_ie,
                       const std::string& gnaCompileTarget,
                       const GNAPluginNS::Config config,
                       bool is_exception_allowed) {
    if (!conv_ie) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "ConvolutionIE node is empty!\n";
        }
        return false;
    }
    size_t batch_size = conv_ie->input_value(0).get_shape()[0];
    if (batch_size != 1) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "topology with layer: " + conv_ie->get_friendly_name() + ", type: " + conv_ie->get_type_name()  +
            ", and batch size(" + std::to_string(batch_size) + ") != 1 not supported";
        }
        return false;
    }
    auto check_dilation = [&](size_t filter_dilation_height, size_t filter_stride_width) -> bool {
        Cnn2D::RangeLimit2D dilation_limit{
            {convDilationHeight, convDilationHeight, "dilation height"},
            {convDilationWidth, convDilationWidth, "dilation width"}};
        std::string error = dilation_limit.GetErrorOrEmpty(filter_dilation_height, filter_stride_width);
        return Cnn2D::AbstractValidator::ValidationSuccesful(is_exception_allowed, error, conv_ie->get_friendly_name(), conv_ie->get_type_name());
    };
    auto input_shape = conv_ie->input_value(0).get_shape();
    auto filter_shape = conv_ie->input_value(1).get_shape();
    if ((4 == filter_shape.size() && filter_shape[2] > 1 && filter_shape[3] > 1) ||
        (4 == input_shape.size() && input_shape[2] > 1 && input_shape[3] > 1)) {
        pass::helper::ConvData conv_data;
        pass::helper::UpdateConvData(conv_ie, conv_data);
        if (GNAPluginNS::GNAConvolutionLayer::isMappableFrom2DTo1D(
            conv_data.input_height, conv_data.input_width, conv_data.input_channel_count,
            conv_data.filter_height, conv_data.filter_width,
            conv_data.filter_stride_height, conv_data.filter_stride_width)) {
            return check_dilation(conv_data.filter_dilation_height, conv_data.filter_dilation_width);
        }
        const auto cnn2dValidatorPtr = Cnn2D::AbstractValidator::Create(gnaCompileTarget);
        if (cnn2dValidatorPtr) {
            return cnn2dValidatorPtr->ValidateCnn2D(conv_ie->get_friendly_name(),
                conv_data.input_height, conv_data.input_width, conv_data.input_channel_count,
                conv_data.filter_height, conv_data.filter_width, conv_data.filter_channel_count,
                conv_data.filter_stride_height, conv_data.filter_stride_width,
                conv_data.filter_dilation_height, conv_data.filter_dilation_width,
                OvGnaTypeIntFromBytes(config.gnaPrecision.size()), is_exception_allowed);
        }
    }
    return check_dilation(conv_ie->get_dilations()[0], conv_ie->get_dilations()[1]);
}

bool is_pooling_supported(const std::shared_ptr<ngraph::opset7::MaxPool> max_pool,
                          const std::string& gnaCompileTarget,
                          bool is_exception_allowed) {
    if (!max_pool) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "MaxPool node is empty!\n";
        }
        return false;
    }
    auto kernels = max_pool->get_kernel();
    if (2 == kernels.size() && kernels[0] > 1 && kernels[1] > 1) {
        const auto cnn2dValidatorPtr = Cnn2D::AbstractValidator::Create(gnaCompileTarget);
        if (cnn2dValidatorPtr) {
            auto strides = max_pool->get_strides();
            return cnn2dValidatorPtr->ValidatePooling2D(max_pool->get_friendly_name(),
                kernels[0], kernels[1], strides[0], strides[1], is_exception_allowed);
        }
    }
    return true;
}

bool is_fc_supported(const std::shared_ptr<ngraph::op::FullyConnected>& fc, bool is_exception_allowed) {
    if (!fc) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "FullyConnected node is empty!\n";
        }
        return false;
    }
    size_t output_batch_size = fc->get_output_shape(0)[0];
    if (output_batch_size > 8) {
        if (is_exception_allowed) {
            THROW_GNA_EXCEPTION << "topology with layer: " + fc->get_friendly_name() + ", type: " + fc->get_type_name() +
            ", and batch size(" + std::to_string(output_batch_size) + ") not supported";
        }
        return false;
    }
    return true;
}
} // namespace

namespace ov {
namespace intel_gna {
namespace ngraph_util {
bool is_op_supported(const std::shared_ptr<ov::Node>& node,
                     const std::string& gnaCompileTarget,
                     const GNAPluginNS::Config config,
                     bool is_exception_allowed) {
    if (ngraph::op::is_parameter(node)) {
        return is_precision_supported(node, supported_parameter_types, is_exception_allowed);
    } else if (ngraph::op::is_constant(node)) {
        return is_precision_supported(node, supported_constant_types, is_exception_allowed);
    } else if (auto conv_ie = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node)) {
        return is_conv_supported(conv_ie, gnaCompileTarget, config, is_exception_allowed);
    } else if (auto fc = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node)) {
        return is_fc_supported(fc, is_exception_allowed);
    } else if (is_pooling(node)) {
        return is_pooling_supported(std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(node), gnaCompileTarget, is_exception_allowed);
    } else if (ngraph::op::is_output(node) ||
               ngraph::op::is_sink(node) ||
               is_eltwise_add(node) ||
               is_eltwise_mul(node) ||
               is_crop_affined(node) ||
               is_activation(node.get()) ||
               is_gna_precision_agnostic(node) || // check concat/split are aligned when transformations will be moved to ngraph
               (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(node) != nullptr) ||
               (std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(node) != nullptr) ||
               (std::dynamic_pointer_cast<ngraph::op::PowerIE>(node) != nullptr) ||
               (std::dynamic_pointer_cast<ngraph::opset9::MatMul>(node) != nullptr)) {
        return true;
    }
    return false;
}
} // namespace ngraph_util
} // namespace intel_gna
} // namespace ov
