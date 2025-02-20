#include <transformations/low_precision/mark_dequantization_subgraph.hpp>

#include <low_precision/common/quantization_granularity_restriction.hpp>
#include <low_precision/convert_subtract_constant.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/convolution_backprop_data.hpp>
#include <low_precision/layer_transformation.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/network_helper.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include "openvino/pass/manager.hpp"

namespace ov {
namespace pass {
namespace device {

class ConvertOpSet1ToDeviceSpecific: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertOpSet1ToDeviceSpecific");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override {
        return true;
    }
};

} // namespace device
} // pass
} // ov

int main() {
std::shared_ptr<ov::Model> model;
ov::pass::Manager manager;
auto pass_config = manager.get_pass_config();
//! [lpt_common]
// check if the function is quantized to ignore LPT transformations for not quantized function to speed up model loading
const bool useLpt = ov::pass::low_precision::LowPrecision::isFunctionQuantized(model);
auto defaultPrecisions =
    useLpt ? ov::pass::low_precision::precision_set::get_int8_support() : std::vector<ov::element::Type>{};
if (useLpt) {
    // disable constant folding on dequantization subgraphs so they can be processed by LPT
    manager.register_pass<ov::pass::MarkDequantization>(defaultPrecisions);
}

// OpenVINO common transformations happen here

if (useLpt) {
    // convert subtract constant to INT8 to prevent unnecessary FP16 to FP32 conversion
    manager.register_pass<ov::pass::low_precision::ConvertSubtractConstant>(defaultPrecisions);
}

// OpenVINO common transformations happen here

if (useLpt) {
    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const std::shared_ptr<const ov::Node> &node) -> bool {
        return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
    });

    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ov::pass::ConvertSubtract>([&defaultPrecisions](const std::shared_ptr<const ov::Node> &node) -> bool {
        return ov::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node, defaultPrecisions);
    });
}

manager.run_passes(model);
//! [lpt_common]

//! [lpt_execution]
using namespace ov::pass::low_precision;
if (useLpt) {
    // Low precision transformations plugin specific configuration: restrictions definition
    auto supportedPrecisions = std::vector<PrecisionsRestriction>({
        PrecisionsRestriction::create<ov::opset1::Convolution>({
            {{0}, {ov::element::u8}},
            {{1}, {ov::element::i8}},
        }),
        PrecisionsRestriction::create<ov::opset1::ConvolutionBackpropData>(
            {{{0}, {ov::element::u8, ov::element::i8}}, {{1}, {ov::element::i8}}}),
        PrecisionsRestriction::create<ov::opset1::GroupConvolution>(
            {{{0}, {ov::element::u8}}, {{1}, {ov::element::i8}}}),
        PrecisionsRestriction::create<ov::opset1::Multiply>({
            {{0}, {ov::element::u8}},
            {{1}, {ov::element::i8}},
        }),
    });

    // Low precision transformations plugin specific configuration: per-tensor quantization operations definition
    auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
        QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0}),
        QuantizationGranularityRestriction::create<ov::opset1::ConvolutionBackpropData>({0})
    });

    // Low precision transformations instantiation and registration in pass manager
    ov::pass::Manager lptManager;
    lptManager.register_pass<ov::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);

    // Low precision transformations plugin specific configuration: transformation callbacks definition
    lptManager.get_pass_config()->set_callback<MarkupPrecisions>([](const std::shared_ptr<const ov::Node>& node) -> bool {
        if (const auto multiply = ov::as_type_ptr<const ov::opset1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(multiply);
        }
        return false;
    });
    lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ov::Node>& node) -> bool {
        return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
    });
    lptManager.get_pass_config()->set_callback<MultiplyToGroupConvolutionTransformation>([](const std::shared_ptr<const ov::Node>& node) -> bool {
        return MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node);
    });

    // Low precision transformations execution
    lptManager.run_passes(model);
}
//! [lpt_execution]

//! [lpt_device]
ov::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ov::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(model);
//! [lpt_device]

return 0;
}

int lpt_supported_precisions() {
std::shared_ptr<ov::Model> model;
ov::pass::Manager manager;

using namespace ov::pass::low_precision;
//! [lpt_supported_precisions]
auto supportedPrecisions = std::vector<PrecisionsRestriction>({
    PrecisionsRestriction::create<ov::opset1::Convolution>({
        {{0}, {ov::element::u8}},
        {{1}, {ov::element::i8}},
    }),
});

ov::pass::Manager lptManager;
lptManager.register_pass<ov::pass::low_precision::LowPrecision>(supportedPrecisions);
lptManager.run_passes(model);
//! [lpt_supported_precisions]

ov::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ov::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(model);

return 0;
}

int per_tensor_quantization() {
std::shared_ptr<ov::Model> model;
//! [per_tensor_quantization]
using namespace ov::pass::low_precision;

const std::vector<PrecisionsRestriction> emptyRestrictions;

auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
    QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0})
});

ov::pass::Manager lptManager;
lptManager.register_pass<ov::pass::low_precision::LowPrecision>(emptyRestrictions, perTensorQuantization);
lptManager.run_passes(model);
//! [per_tensor_quantization]

return 0;
}

int asymmetric_quantization(const std::vector<ov::element::Type>& defaultPrecisions) {
std::shared_ptr<ov::Model> model;
ov::pass::Manager manager;
auto pass_config = manager.get_pass_config();


//! [asymmetric_quantization]
using namespace ov::pass::low_precision;
ov::pass::Manager lptManager;

lptManager.register_pass<ov::pass::low_precision::LowPrecision>();
lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ov::Node>& node) -> bool {
    return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
});
lptManager.run_passes(model);
//! [asymmetric_quantization]

return 0;
}

int lpt_markup_pipeline() {
std::shared_ptr<ov::Model> model;
ov::pass::Manager manager;

using namespace ov::pass::low_precision;
//! [lpt_markup_pipeline]
auto supportedPrecisions = std::vector<PrecisionsRestriction>({
    PrecisionsRestriction::create<ov::opset1::Convolution>({
        {{0}, {ov::element::u8}},
        {{1}, {ov::element::i8}},
    }),
});

auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
    QuantizationGranularityRestriction::create<ov::opset1::Convolution>({0})
});

ov::pass::Manager lptManager;
lptManager.register_pass<ov::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);
lptManager.run_passes(model);
//! [lpt_markup_pipeline]

ov::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ov::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(model);

return 0;
}
