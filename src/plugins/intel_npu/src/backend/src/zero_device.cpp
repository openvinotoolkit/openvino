// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/common_optimizations/reduce_reshape_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/rms_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_avgpool_downgrade.hpp>
#include <transformations/op_conversions/convert_broadcast_to_tiles.hpp>
#include <transformations/op_conversions/convert_convertlike.hpp>
#include <transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp>
#include <transformations/op_conversions/convert_gather_upgrade.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include <transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp>
#include <transformations/op_conversions/convert_maxpool_downgrade.hpp>
#include <transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_pad12_downgrade.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_9.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>
#include <transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/op_conversions/convert_slice_to_strided_slice.hpp>
#include <transformations/op_conversions/convert_softmax_upgrade.hpp>
#include <transformations/op_conversions/convert_topk11_downgrade.hpp>
#include <transformations/op_conversions/detection_output_downgrade.hpp>
#include <transformations/op_conversions/einsum_decomposition.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/op_conversions/log_softmax_decomposition.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/utils/utils.hpp>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "zero_executor.hpp"
#include "zero_host_tensor.hpp"
#include "zero_infer_request.hpp"
#include "zero_remote_tensor.hpp"
#include "zero_utils.hpp"

using namespace intel_npu;

namespace {

void runModelPasses(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    ov::element::TypeVector decompression_precisions{
        ov::element::u4,
        ov::element::i4,
        ov::element::nf4,
        ov::element::u8,
        ov::element::i8,
    };
    manager.register_pass<ov::pass::MarkDequantizationSubgraph>(decompression_precisions, /*fold_subtract_const=*/true);
    manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3>();
    manager.register_pass<ov::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
    manager.register_pass<ov::pass::ConvertPad12ToPad1>();
    manager.register_pass<ov::pass::ConstantFolding>();
    // MOCTransformations contain StridedSliceOptimization transformation,
    // so we must call SliceToStridedSlice before MOCTransformations call
    manager.register_pass<ov::pass::SliceToStridedSlice>(true);
    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    manager.register_pass<ov::pass::MOCTransformations>(true, false);

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::PadFusionConvolution>();
    pass_config->disable<ov::pass::PadFusionGroupConvolution>();
    pass_config->disable<ov::pass::MVNFusionWithConstantsInside>();
    pass_config->disable<ov::pass::PullThroughReduce>();
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();

    // NMS conversion passes
    manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();

    auto static_shape = manager.register_pass<ov::pass::GraphRewrite>();
    static_shape->add_matcher<ov::pass::ConvertNMS9ToNMSIEInternal>();
    static_shape->set_name("ov::pass::CommonStaticShape");

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    common_fusions->add_matcher<ov::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ov::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ov::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ov::pass::TransposeToReshape>();
    common_fusions->add_matcher<ov::pass::RMSFusion>();
    common_fusions->set_name("ov::pass::CommonFusions");

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    decomp->add_matcher<ov::pass::Gelu7Downgrade>();
    decomp->add_matcher<ov::pass::BidirectionalGRUSequenceDecomposition>();
    decomp->add_matcher<ov::pass::BidirectionalRNNSequenceDecomposition>();
    decomp->add_matcher<ov::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ov::pass::ConvertConvertLike>();
    decomp->add_matcher<ov::pass::BatchNormDecomposition>();
    decomp->add_matcher<ov::pass::EinsumDecomposition>();
    decomp->add_matcher<ov::pass::DropoutWithRandomUniformReplacer>();
    decomp->add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    decomp->set_name("ov::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ov::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ov::pass::LinOpSequenceFusion>();
    manager.register_pass<ov::pass::UnrollIf>();

    auto conv_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    conv_fusions->add_matcher<ov::pass::ConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::ConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyConvolutionBackpropDataFusion>();
    conv_fusions->add_matcher<ov::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    conv_fusions->set_name("ov::pass::ConvFusions");

    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::ConvertGather1ToGather7>();
    manager.register_pass<ov::pass::ConvertGather7ToGather8>();
    manager.register_pass<ov::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ov::pass::ConvertMaxPool14ToMaxPool8>();
    manager.register_pass<ov::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ov::pass::ConvertAvgPool14ToAvgPool1>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.register_pass<ov::pass::ConvertDetectionOutput8ToDetectionOutput1>();
    manager.register_pass<ov::pass::ConvertShapeOf3>();

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ov::pass::StridesOptimization>();
    manager.register_pass<ov::pass::ConvertSoftMax1ToSoftMax8>();
    manager.run_passes(model);
}

}  // namespace

ZeroDevice::ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      _graph_ddi_table_ext(_initStructs->getGraphDdiTable()),
      log("ZeroDevice", Logger::global().level()) {
    log.debug("ZeroDevice::ZeroDevice init");
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_initStructs->getDevice(), &device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility. For any other error, we just fall-back to device ID to assure backwards compatibilty with even
    // older drivers
    pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_initStructs->getDevice(), &pci_properties);
    if (ZE_RESULT_SUCCESS == retpci) {
        // windows driver specific backwards compatibility
        if (pci_properties.address.device == 0) {
            log.warning("PCI information not available in driver. Falling back to deviceId");
            pci_properties.address.device = device_properties.deviceId;
        }
    } else {
        // general backwards compatibility
        log.warning("PCI information not available in driver. Falling back to deviceId");
        pci_properties.address.device = device_properties.deviceId;
    }

    /// Calculate and store device GOPS with formula: frequency * number of tiles * ops per tile
    /// cross-OS backwards compatibilty: only calculate gops if driver supports it (version>x)
    uint32_t gops_support_drv_version = UINT32_MAX;
#if defined(_WIN32) || defined(__CYGWIN__)
    gops_support_drv_version = 2465;  /// Windows driver version which supports Gops calculations
#else                                 // _WIN32 || __CYGWIN__
    gops_support_drv_version = 1715354569;  /// Linux driver version which supports Gops calculations
#endif                                // _WIN32 || __CYGWIN__
    if (_initStructs->getDriverVersion() >= gops_support_drv_version) {
        float gops = (device_properties.coreClockRate / powf(1000, 3)) * device_properties.numSlices *
                     device_properties.physicalEUSimdWidth;
        device_gops[ov::element::f32] = 0;
        device_gops[ov::element::u8] = gops;
        device_gops[ov::element::i8] = gops;
        device_gops[ov::element::f16] = 0.5f * gops;
    }

    std::vector<ze_command_queue_group_properties_t> command_group_properties;
    uint32_t command_queue_group_count = 0;
    // Discover all command queue groups
    zeroUtils::throwOnFail(
        "zeDeviceGetCommandQueueGroupProperties",
        zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(), &command_queue_group_count, nullptr));

    log.debug("ZeroDevice::ZeroDevice - resize command_queue_group_count");
    command_group_properties.resize(command_queue_group_count);

    for (auto& prop : command_group_properties) {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }

    zeroUtils::throwOnFail("zeDeviceGetCommandQueueGroupProperties",
                           zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(),
                                                                  &command_queue_group_count,
                                                                  command_group_properties.data()));

    // Find the corresponding command queue group.
    log.debug("ZeroDevice::ZeroDevice - findGroupOrdinal");
    _group_ordinal = zeroUtils::findGroupOrdinal(command_group_properties, device_properties);
    log.debug("ZeroDevice::ZeroDevice - init completed");
}

std::shared_ptr<IExecutor> ZeroDevice::createExecutor(
    const std::shared_ptr<const NetworkDescription>& networkDescription,
    const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Device::createExecutor");
    return std::make_shared<ZeroExecutor>(_initStructs, networkDescription, config, _group_ordinal);
}

std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> ZeroDevice::runInit(
    const std::shared_ptr<IExecutor>& initExecutor,
    const std::shared_ptr<const ov::Model>& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const Config& config) {
    const auto zeroInitExecutor = static_cast<const ZeroExecutor*>(initExecutor.get());
    std::unordered_map<size_t, TensorData> constantIdToTensorData;
    std::vector<std::optional<TensorData>> inputTensorsData;
    std::vector<std::optional<TensorData>> outputTensorsData;
    std::vector<std::shared_ptr<ov::ITensor>> inputHostTensors;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> outputHostTensors;
    std::vector<std::vector<uint8_t>> dummy;

    auto clonedModel = model->clone();
    runModelPasses(clonedModel);

    // Match the inputs of the "init" model with the Constant nodes of the original model
    size_t constantIndex = 0;
    for (auto&& node : clonedModel->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        const auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        const void* address = constantNode->get_data_ptr();
        const size_t size = constantNode->get_byte_size();
        constantIdToTensorData.emplace(constantIndex++, TensorData{address, size});
    }

    for (const auto& descriptor : zeroInitExecutor->get_input_descriptors()) {
        size_t id = std::stoi(std::string(descriptor.info.name).substr(INIT_INPUT_WEIGHTS_PREFIX.length()));
        OPENVINO_ASSERT(constantIdToTensorData.count(id), "Mismatch between weights IDs and parsed inputs");
        const ov::SoPtr<ov::ITensor> hostTensor =
            createHostTensor(context._ptr,
                             zeroUtils::getOVPrecision(descriptor.info.devicePrecision),
                             zeroUtils::getOVShape(descriptor.info),
                             config);

        OPENVINO_ASSERT(constantIdToTensorData.at(id).size == hostTensor->get_byte_size(),
                        "Byte size mismatch for ",
                        descriptor.info.name);
        std::memcpy(hostTensor->data(), constantIdToTensorData.at(id).mem, hostTensor->get_byte_size());

        inputTensorsData.push_back(TensorData{hostTensor->data(), hostTensor->get_byte_size()});
        inputHostTensors.push_back(hostTensor._ptr);
    }

    for (const auto& descriptor : zeroInitExecutor->get_output_descriptors()) {
        const ov::SoPtr<ov::ITensor> hostTensor =
            createHostTensor(context._ptr,
                             zeroUtils::getOVPrecision(descriptor.info.devicePrecision),
                             zeroUtils::getOVShape(descriptor.info),
                             config);
        outputTensorsData.push_back(TensorData{hostTensor->data(), hostTensor->get_byte_size()});
        outputHostTensors.emplace(
            std::string(descriptor.info.debug_friendly_name).substr(INIT_OUTPUT_WEIGHTS_PREFIX.length()),
            hostTensor._ptr);
    }

    auto progilingPool = zeroProfiling::ProfilingPool(zeroInitExecutor->graph(),
                                                      zeroProfiling::POOL_SIZE,
                                                      zeroInitExecutor->getInitStructs()->getProfilingDdiTable());
    auto profilingQuery = zeroProfiling::ProfilingQuery(0,
                                                        zeroInitExecutor->getInitStructs()->getDevice(),
                                                        zeroInitExecutor->getInitStructs()->getProfilingDdiTable());
    const auto pipeline = std::make_unique<Pipeline>(config,
                                                     initExecutor,
                                                     progilingPool,
                                                     profilingQuery,
                                                     nullptr,
                                                     inputTensorsData,
                                                     outputTensorsData,
                                                     /*numberOfCommandLists*/ 1);
    pipeline->push();
    pipeline->pull();

    return outputHostTensors;
}

std::string ZeroDevice::getName() const {
//    KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D
#define NPU_4000_DEVICE_ID   0x643E

    std::string name;
    switch (device_properties.deviceId) {
    case NPU_3720_P_DEVICE_ID:
    case NPU_3720_S_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU3720;
        break;
    case NPU_4000_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU4000;
        break;
    default:
        name = ov::intel_npu::Platform::AUTO_DETECT;
    }

    return name;
}

std::string ZeroDevice::getFullDeviceName() const {
    return device_properties.name;
}

IDevice::Uuid ZeroDevice::getUuid() const {
    Uuid uuid{};
    static_assert(sizeof(device_properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match intel_npu::Uuid::uuid size");

    std::copy(std::begin(device_properties.uuid.id), std::end(device_properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

uint32_t ZeroDevice::getSubDevId() const {
    return device_properties.subdeviceId;
}

uint32_t ZeroDevice::getMaxNumSlices() const {
    return device_properties.numSlices;
}

uint64_t ZeroDevice::getAllocMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext.pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.allocated;
}

uint64_t ZeroDevice::getTotalMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext.pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.total;
}

ov::device::PCIInfo ZeroDevice::getPciInfo() const {
    return ov::device::PCIInfo{pci_properties.address.domain,
                               pci_properties.address.bus,
                               pci_properties.address.device,
                               pci_properties.address.function};
}

std::map<ov::element::Type, float> ZeroDevice::getGops() const {
    return device_gops;
}

ov::device::Type ZeroDevice::getDeviceType() const {
    return ov::device::Type::INTEGRATED;
}

std::shared_ptr<SyncInferRequest> ZeroDevice::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const std::shared_ptr<IExecutor>& executor,
    const Config& config) {
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, executor, config);
}

ov::SoPtr<ov::IRemoteTensor> ZeroDevice::createRemoteTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                            const ov::element::Type& element_type,
                                                            const ov::Shape& shape,
                                                            const Config& config,
                                                            ov::intel_npu::TensorType tensor_type,
                                                            ov::intel_npu::MemType mem_type,
                                                            const void* mem) {
    return {std::make_shared<
        ZeroRemoteTensor>(context, _initStructs, element_type, shape, config, tensor_type, mem_type, mem)};
};

ov::SoPtr<ov::ITensor> ZeroDevice::createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape,
                                                    const Config& config) {
    return {std::make_shared<ZeroHostTensor>(context, _initStructs, element_type, shape, config)};
};
