#include <ie_core.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>

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

namespace ngraph {
namespace pass {
namespace device {

class ConvertOpSet1ToDeviceSpecific: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        return true;
    }
};

} // namespace device
} // pass
} // ngraph

int main() {
std::shared_ptr<ov::Model> nGraphFunc;
ngraph::pass::Manager manager;
auto pass_config = manager.get_pass_config();
//! [lpt_common]
// check if the function is quantized to ignore LPT transformations for not quantized function to speed up model loading
const bool useLpt = ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(nGraphFunc);
auto defaultPrecisions =
    useLpt ? ngraph::pass::low_precision::precision_set::int8_support : std::vector<ov::element::Type>{};
if (useLpt) {
    // disable constant folding on constant subgraph to use the subgraph for LPT
    manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(defaultPrecisions);
}

// nGraph common transformations happen here

if (useLpt) {
    // convert subtract constant to INT8 to prevent unnecessary FP16 to FP32 conversion
    manager.register_pass<ngraph::pass::low_precision::ConvertSubtractConstant>(defaultPrecisions);
}

// nGraph common transformations happen here

if (useLpt) {
    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([&defaultPrecisions](const std::shared_ptr<const ngraph::Node> &node) -> bool {
        return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node, defaultPrecisions);
    });

    // convert not supported cases FakeQuantize -> Convert -> Convert -> Subtract -> Multiply to a single FakeQuantize
    pass_config->set_callback<ngraph::pass::ConvertSubtract>([&defaultPrecisions](const std::shared_ptr<const ngraph::Node> &node) -> bool {
        return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node, defaultPrecisions);
    });
}

manager.run_passes(nGraphFunc);
//! [lpt_common]

//! [lpt_execution]
using namespace ngraph::pass::low_precision;
if (useLpt) {
    // Low precision transformations plugin specific configuration: restrictions definition
    auto supportedPrecisions = std::vector<PrecisionsRestriction>({
        PrecisionsRestriction::create<ngraph::opset1::Convolution>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}},
        }),
        PrecisionsRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
            {0, {ngraph::element::u8, ngraph::element::i8}},
            {1, {ngraph::element::i8}}
        }),
        PrecisionsRestriction::create<ngraph::opset1::GroupConvolution>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}}
        }),
        PrecisionsRestriction::create<ngraph::opset1::Multiply>({
            {0, {ngraph::element::u8}},
            {1, {ngraph::element::i8}},
        }),
    });

    // Low precision transformations plugin specific configuration: per-tensor quantization operations definition
    auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
        QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0}),
        QuantizationGranularityRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0})
    });

    // Low precision transformations instantiation and registration in pass manager
    ngraph::pass::Manager lptManager;
    lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);

    // Low precision transformations plugin specific configuration: transformation callbacks definition
    lptManager.get_pass_config()->set_callback<MarkupPrecisions>([](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        if (const auto multiply = std::dynamic_pointer_cast<const ngraph::opset1::Multiply>(node)) {
            return !MultiplyToGroupConvolutionTransformation::canBeTransformedToGroupConvolution(multiply);
        }
        return false;
    });
    lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
    });
    lptManager.get_pass_config()->set_callback<MultiplyToGroupConvolutionTransformation>([](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        return MultiplyToGroupConvolutionTransformation::isDynamicOrScalar(node);
    });

    // Low precision transformations execution
    lptManager.run_passes(nGraphFunc);
}
//! [lpt_execution]

//! [lpt_device]
ngraph::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ngraph::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(nGraphFunc);
//! [lpt_device]

return 0;
}

int lpt_supported_precisions() {
std::shared_ptr<ov::Model> nGraphFunc;
ngraph::pass::Manager manager;

using namespace ngraph::pass::low_precision;
//! [lpt_supported_precisions]
auto supportedPrecisions = std::vector<PrecisionsRestriction>({
    PrecisionsRestriction::create<ngraph::opset1::Convolution>({
        {0, {ngraph::element::u8}},
        {1, {ngraph::element::i8}},
    }),
});

ngraph::pass::Manager lptManager;
lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions);
lptManager.run_passes(nGraphFunc);
//! [lpt_supported_precisions]

ngraph::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ngraph::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(nGraphFunc);

return 0;
}

int per_tensor_quantization() {
std::shared_ptr<ov::Model> nGraphFunc;
//! [per_tensor_quantization]
using namespace ngraph::pass::low_precision;

const std::vector<PrecisionsRestriction> emptyRestrictions;

auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
    QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0})
});

ngraph::pass::Manager lptManager;
lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(emptyRestrictions, perTensorQuantization);
lptManager.run_passes(nGraphFunc);
//! [per_tensor_quantization]

return 0;
}

int asymmetric_quantization(const std::vector<ngraph::element::Type>& defaultPrecisions) {
std::shared_ptr<ov::Model> nGraphFunc;
ngraph::pass::Manager manager;
auto pass_config = manager.get_pass_config();


//! [asymmetric_quantization]
using namespace ngraph::pass::low_precision;
ngraph::pass::Manager lptManager;

lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>();
lptManager.get_pass_config()->set_callback<ConvolutionBackpropDataTransformation>([&defaultPrecisions](const std::shared_ptr<const ngraph::Node>& node) -> bool {
    return LayerTransformation::isAsymmetricQuantization(node, defaultPrecisions) || WeightableLayerTransformation::isAsymmetricOnWeights(node);
});
lptManager.run_passes(nGraphFunc);
//! [asymmetric_quantization]

return 0;
}

int lpt_markup_pipeline() {
std::shared_ptr<ov::Model> nGraphFunc;
ngraph::pass::Manager manager;

using namespace ngraph::pass::low_precision;
//! [lpt_markup_pipeline]
auto supportedPrecisions = std::vector<PrecisionsRestriction>({
    PrecisionsRestriction::create<ngraph::opset1::Convolution>({
        {0, {ngraph::element::u8}},
        {1, {ngraph::element::i8}},
    }),
});

auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
    QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0})
});

ngraph::pass::Manager lptManager;
lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);
lptManager.run_passes(nGraphFunc);
//! [lpt_markup_pipeline]

ngraph::pass::Manager deviceSpecificManager;
deviceSpecificManager.register_pass<ngraph::pass::device::ConvertOpSet1ToDeviceSpecific>();
deviceSpecificManager.run_passes(nGraphFunc);

return 0;
}
