// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_ie.hpp"

#include <ie_blob.h>

#include <algorithm>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "blob_factory.hpp"
#include <legacy/ie_ngraph_utils.hpp>
#include <precision_utils.h>
#include "ngraph/util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

#include "transformations/rt_info/generic_ie_convert_precision.hpp"

constexpr ::ngraph::NodeTypeInfo ngraph::op::GenericIE::type_info;

void ngraph::op::GenericIE::addExtension(std::shared_ptr<const ngraph::Function> func,
                                         const InferenceEngine::IShapeInferExtensionPtr& ext) {
    NodeVector nodes;

    for (auto r : func->get_results())
        nodes.emplace_back(r);
    for (auto param : func->get_parameters())
        nodes.emplace_back(param);

    traverse_nodes(nodes, [&](std::shared_ptr<Node> op) {
        if (auto generic = std::dynamic_pointer_cast<GenericIE>(op)) {
            generic->addExtension(ext);
        }
        if (auto ti = std::dynamic_pointer_cast<ngraph::op::TensorIterator>(op)) {
            addExtension(ti->get_body(), ext);
        }
    });
}

void ngraph::op::GenericIE::addExtension(const InferenceEngine::IShapeInferExtensionPtr& ext) {
    extensions.emplace_back(ext);
}

std::vector<InferenceEngine::IShapeInferExtensionPtr> ngraph::op::GenericIE::getExtensions(std::shared_ptr<const ngraph::Function> func) {
    for (auto& op : func->get_ops()) {
        if (auto generic = std::dynamic_pointer_cast<GenericIE>(op)) {
            return generic->getExtensions();
        }
    }
    return {};
}

std::vector<InferenceEngine::IShapeInferExtensionPtr> ngraph::op::GenericIE::getExtensions() {
    return extensions;
}

ngraph::op::GenericIE::GenericIE(const ngraph::OutputVector& inputs,
                                 const std::map<std::string, InferenceEngine::Parameter>& params_,
                                 const std::string type_, const std::vector<PortIE>& outputs_)
    : Op(inputs), params(params_), outputs(outputs_), type(type_), initialized(0) {
    constructor_validate_and_infer_types();
    auto & rt_info = get_rt_info();
    rt_info[VariantWrapper<GenericIEConvertPrecision>::type_info.name] =
            std::make_shared<VariantWrapper<GenericIEConvertPrecision>>(
                    GenericIEConvertPrecision([this](const element::Type & m_type) {
                        this->convert_precision(m_type);
                    }));
}

// Copy from net_pass.h
template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
void inline convertArrayPrecision(typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type *dst,
                                  const typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type *src,
                                  size_t nelem) {
    using dst_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    for (size_t i = 0; i < nelem; i++) {
        dst[i] = static_cast<dst_type>(src[i]);
    }
}

template<>
void inline
convertArrayPrecision<InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32>(float *dst, const short *src,
                                                                                          size_t nelem) {
    uint16_t a = *reinterpret_cast<const uint16_t *>(src);
    InferenceEngine::PrecisionUtils::f16tof32Arrays(dst, src, nelem, 1.0f, 0.0f);
}

template<InferenceEngine::Precision::ePrecision PREC_FROM, InferenceEngine::Precision::ePrecision PREC_TO>
InferenceEngine::Blob::Ptr inline convertBlobPrecision(const InferenceEngine::Blob::Ptr &blob) {
    using from_d_type = typename InferenceEngine::PrecisionTrait<PREC_FROM>::value_type;
    using to_d_type = typename InferenceEngine::PrecisionTrait<PREC_TO>::value_type;

    auto tensor_desc = blob->getTensorDesc();
    InferenceEngine::Blob::Ptr new_blob = InferenceEngine::make_shared_blob<to_d_type>(
            InferenceEngine::TensorDesc{PREC_TO, tensor_desc.getDims(), tensor_desc.getLayout()});
    new_blob->allocate();
    auto target = new_blob->buffer().as<to_d_type *>();
    auto source = blob->buffer().as<from_d_type *>();
    convertArrayPrecision<PREC_FROM, PREC_TO>(target, source, blob->size());
    return new_blob;
}

template<InferenceEngine::Precision::ePrecision targetPRC>
InferenceEngine::Blob::Ptr inline copyBlobWithCast(const InferenceEngine::Blob::Ptr &blob) {
    if (blob->getTensorDesc().getPrecision() == targetPRC) {
        return blob;
    }
    InferenceEngine::Blob::Ptr newBlob;
    switch (blob->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::FP32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::FP16:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::FP16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I16:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::I16, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I8:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::I8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::U8:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::U8, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::I32:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::I32, targetPRC>(blob);
            break;
        case InferenceEngine::Precision::BOOL:
            newBlob = convertBlobPrecision<InferenceEngine::Precision::BOOL, targetPRC>(blob);
            break;
        default:
            THROW_IE_EXCEPTION << "Conversion from blob with precision " << blob->getTensorDesc().getPrecision().name()
                               << " not implemented yet!";
    }
    return newBlob;
}


void ngraph::op::GenericIE::convert_precision(const element::Type & el_type) {
    for (auto & param : params) {
        if (param.second.is<InferenceEngine::Blob::Ptr>()) {
            auto blob = param.second.as<InferenceEngine::Blob::Ptr>();
            InferenceEngine::Blob::Ptr newBlob;
            switch (el_type) {
                case ngraph::element::Type_t::f32:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::FP32>(blob);
                    break;
                case ngraph::element::Type_t::f16:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::FP16>(blob);
                    break;
                case ngraph::element::Type_t::i16:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::I16>(blob);
                    break;
                case ngraph::element::Type_t::i8:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::I8>(blob);
                    break;
                case ngraph::element::Type_t::u8:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::U8>(blob);
                    break;
                case ngraph::element::Type_t::i32:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::I32>(blob);
                    break;
                case ngraph::element::Type_t::boolean:
                    newBlob = copyBlobWithCast<InferenceEngine::Precision::BOOL>(blob);
                    break;
                default:
                    THROW_IE_EXCEPTION << "Conversion from blob with precision " << blob->getTensorDesc().getPrecision().name()
                                       << " not implemented yet!";
            }
            param.second = newBlob;
        }
    }
}

std::shared_ptr<ngraph::Node> ngraph::op::GenericIE::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto genNode = std::make_shared<GenericIE>(new_args, params, type, outputs);
    genNode->extensions = extensions;
    genNode->reshape = reshape;
    return genNode;
}

void ngraph::op::GenericIE::validate_and_infer_types() {
    // This function returns precision based on existing precision and
    // precision that was set in outputs vector
    auto get_precision = [this](const size_t index) -> element::Type {
        if (index >= get_output_size() ||
            get_output_element_type(index) == element::dynamic ||
            get_output_element_type(index) == element::undefined) {
            return InferenceEngine::details::convertPrecision(outputs[index].precision);
        }
        return get_output_element_type(index);
    };
    // Try to find extension with shape inference implementation and apply it
    for (const auto& ext : extensions) {
        IE_SUPPRESS_DEPRECATED_START
        InferenceEngine::IShapeInferImpl::Ptr impl;
        InferenceEngine::StatusCode ret = ext->getShapeInferImpl(impl, type.c_str(), nullptr);
        if (ret != InferenceEngine::StatusCode::OK || !impl) continue;

        std::vector<InferenceEngine::Blob::CPtr> inputs;
        std::map<std::string, std::string> parameters;
        std::map<std::string, InferenceEngine::Blob::Ptr> blobs;
        std::vector<InferenceEngine::SizeVector> outShapes;

        for (uint64_t i = 0; i < get_input_size(); i++) {
            PartialShape this_input_shape = get_input_partial_shape(i);

            if (!this_input_shape.is_static()) {
                // Set dynamic output shapes if input shapes are not defined
                for (size_t output_index = 0; output_index < outputs.size(); output_index++) {
                    set_output_type(output_index, get_precision(output_index), PartialShape::dynamic());
                }
                return;
            }

            Shape this_ishape = get_input_shape(i);
            InferenceEngine::SizeVector dims = this_ishape;
            InferenceEngine::Blob::Ptr input = make_blob_with_precision(InferenceEngine::TensorDesc(
                InferenceEngine::details::convertPrecision(get_input_element_type(i)), dims,
                InferenceEngine::TensorDesc::getLayoutByDims(dims)));
            inputs.emplace_back(input);
        }

        for (const auto& attr : params) {
            if (attr.second.is<std::string>()) {
                parameters[attr.first] = attr.second.as<std::string>();
            } else if (attr.second.is<InferenceEngine::Blob::CPtr>()) {
                auto cBlob = attr.second.as<InferenceEngine::Blob::CPtr>();
                auto wBlob = std::const_pointer_cast<InferenceEngine::Blob>(cBlob);
                blobs[attr.first] = wBlob;
            } else if (attr.second.is<InferenceEngine::Blob::Ptr>()) {
                auto wBlob = attr.second.as<InferenceEngine::Blob::Ptr>();
                blobs[attr.first] = wBlob;
            } else {
                THROW_IE_EXCEPTION << "Generic node for layer " << get_friendly_name() << " with type " << type
                                   << " has incorrect parameter " << attr.first << "!";
            }
        }

        // WA: shape infer has to know number of outputs
        if ((type == "ExperimentalDetectronROIFeatureExtractor" || type == "ExperimentalDetectronDetectionOutput")
                && parameters.find("num_outputs") == parameters.end()) {
            parameters["num_outputs"] = std::to_string(outputs.size());
        }

        ret = impl->inferShapes(inputs, parameters, blobs, outShapes, nullptr);
        IE_SUPPRESS_DEPRECATED_END

        if (ret != InferenceEngine::StatusCode::OK || outShapes.size() != outputs.size()) continue;

        for (size_t output_index = 0; output_index < outputs.size(); output_index++) {
            set_output_type(output_index, get_precision(output_index), Shape(outShapes[output_index]));
        }
        return;
    }

    // Extensions are not loaded when we create nGraph function
    // First call: create node
    if (initialized < 1) {
        if ((type == "ExperimentalDetectronROIFeatureExtractor" || type == "ExperimentalDetectronDetectionOutput")
                && outputs.size() < 2) {
            // Add fake port
            PortIE port;
            port.precision = InferenceEngine::Precision::FP32;
            outputs.emplace_back(port);
        }
        if (outputs.size())
            set_output_size(outputs.size());
        for (size_t output_index = 0; output_index < outputs.size(); output_index++) {
            set_output_type(output_index, get_precision(output_index), Shape(outputs[output_index].dims));
        }
        initialized++;
    } else if (reshape) {
        THROW_IE_EXCEPTION << "IShapeInferExtension wasn't registered for node " << get_friendly_name()
                           << " with type " << type;
    }
}
