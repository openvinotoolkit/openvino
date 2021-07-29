# Inference Engine API Changes History {#openvino_docs_IE_DG_API_Changes}

The sections below contain detailed list of changes made to the Inference Engine API in recent releases.

## 2021.4

### New API

* InferenceEngine::Core::LoadNetwork(modelPath, deviceName, config) simplified API to read and load network in one call

### Deprecated API

 **InferenceEngine::Parameter**

 * InferenceEngine::Parameter(const std::shared_ptr<ngraph::Variant>&)
 * InferenceEngine::Parameter(std::shared_ptr<ngraph::Variant>& var)
 * std::shared_ptr<ngraph::Variant> InferenceEngine::Parameter::asVariant() const
 * InferenceEngine::Parameter::operator std::shared_ptr<ngraph::Variant>() const

 **GPU plugin configuration keys**
 * KEY_CLDNN_NV12_TWO_INPUTS GPU plugin option. Use KEY_GPU_NV12_TWO_INPUTS instead
 * KEY_CLDNN_PLUGIN_PRIORITY GPU plugin option. Use KEY_GPU_PLUGIN_PRIORITY instead
 * KEY_CLDNN_PLUGIN_THROTTLE GPU plugin option. Use KEY_GPU_PLUGIN_THROTTLE instead
 * KEY_CLDNN_MEM_POOL GPU plugin option
 * KEY_CLDNN_GRAPH_DUMPS_DIR GPU plugin option
 * KEY_CLDNN_SOURCES_DUMPS_DIR GPU plugin option
 * KEY_DUMP_KERNELS GPU plugin option
 * KEY_TUNING_MODE GPU plugin option
 * KEY_TUNING_FILE GPU plugin option

 **InferenceEngine::IInferRequest**
 * IInferRequest interface is deprecated, use InferRequest wrapper:
  * Constructor for InferRequest from IInferRequest:: Ptr is deprecated
  * Cast operator for InferRequest to IInferRequest shared pointer is deprecated

 **InferenceEngine::ICNNNetwork**
 * ICNNNetwork interface is deprecated by means of deprecation of all its methods, use CNNNetwork wrapper
  * CNNNetwork methods working with ICNNNetwork are deprecated:
  * Cast to ICNNNetwork shared pointer
  * Cast to reference to ICNNNetwork interface
  * Constructor from ICNNNetwork shared pointer

 **InferenceEngine::IExecutableNetwork**
 * IExecutableNetwork is deprecated, use ExecutableNetwork wrappers:
  * Constructor of ExecutableNetwork from IExecutableNetwork shared pointer is deprecated
 * The following ExecutableNetwork methods are deprecated:
  * ExecutableNetwork::reset
  * Cast operator to IExecutableNetwork shared pointer
  * ExecutableNetwork::CreateInferRequestPtr - use ExecutableNetwork::CreateInferRequest instead

 **Extensions API**
 * InferenceEngine::make_so_pointer which is used to create Extensions library is replaced by std::make_shared<Extension>(..)
 * InferenceEngine::IExtension::Release is deprecated with no replacement
 * Use IE_DEFINE_EXTENSION_CREATE_FUNCTION helper macro instead of explicit declaration of CreateExtension function, which create extension.

 **Other changes**
 * Version::ApiVersion structure is deprecated, Inference Engine does not have API version anymore
 * LowLatency - use lowLatency2 instead
 * CONFIG_KEY(DUMP_EXEC_GRAPH_AS_DOT) - use InferenceEngine::ExecutableNetwork::GetExecGraphInfo::serialize() instead
 * Core::ImportNetwork with no device - pass device name explicitly.
 * details::InferenceEngineException - use InferenceEngine::Exception and its derivatives instead.

## 2021.3

### New API

 * InferenceEngine::InferRequest::Cancel to cancel inference request execution
 * InferenceEngine::Layout::HWC to support HWC layout for input or output blobs
 * InferenceEngine::Precision::F64 data precision for f64 data type
 * InferenceEngine::CNNNetwork::getOVNameForTensor to map frameworks tensor names to OpenVINO internal tensor names

### Deprecated API

 * InferenceEngine::IVariableState interface is deprecated, use InferenceEngine::VariableState wrapper

## 2021.2

### New API

 **State API**

 * InferenceEngine::InferRequest::QueryState query state value of network on current infer request
 * InferenceEngine::IVariableState class instead of IMemoryState (rename)
 * InferenceEngine::IVariableState::GetState instead of IMemoryState::GetLastState (rename)

 **BatchedBlob** - represents a InferenceEngine::BatchedBlob containing other blobs - one per batch.

 **Transformations API** - added a new header `ie_transformations.hpp` which contains transformations for InferenceEngine::CNNNetwork object. Such transformations can be called prior to loading network for compilation for particular device:

 * InferenceEngine::LowLatency

### Deprecated API

 **State API**

 * InferenceEngine::ExecutableNetwork::QueryState - use InferenceEngine::InferRequest::QueryState
 * InferenceEngine::IVariableState::GetLastState - use InferenceEngine::IVariableState::GetState

## 2021.1

### Deprecated API

 **Utility functions to convert Unicode paths**

 * InferenceEngine::stringToFileName - use OS-specific native conversion functions
 * InferenceEngine::fileNameToString - use OS-specific native conversion functions

### Removed API

 **Plugin API:**

 * InferenceEngine::InferencePlugin C++ plugin wrapper class
 * InferenceEngine::IInferencePlugin plugin interface
 * InferenceEngine::PluginDispatcher class
 * InferenceEngine::InferenceEnginePluginPtr typedef
 * InferenceEngine::ICNNNetReader reader interface
 * InferenceEngine::CNNNetReader class

 **Extensibility API:**

 * InferenceEngine::ILayerImplFactory class
 * InferenceEngine::IShapeInferImpl class
 * InferenceEngine::IShapeInferExtension class
 * InferenceEngine::IExtension::getFactoryFor(ILayerImplFactory\*& factory, const CNNLayer\* cnnLayer, ResponseDesc\* resp) noexcept method
 * InferenceEngine::IExtension::getPrimitiveTypes(char\*\*& types, unsigned int& size, ResponseDesc\* resp) noexcept method
 * InferenceEngine::ShapeInferImpl class
 * InferenceEngine::Extension::getFactoryFor(ILayerImplFactory\*& factory, const CNNLayer\* cnnLayer, ResponseDesc\* resp) noexcept method
 * InferenceEngine::Extension::getPrimitiveTypes(char\*\*& types, unsigned int& size, ResponseDesc\* resp) noexcept method

 **Network API:**

 * InferenceEngine::details::CNNNetworkIterator class
 * InferenceEngine::CNNNetwork::getPrecision() const method
 * InferenceEngine::CNNNetwork::getLayerByName(const char\* layerName) const method
 * InferenceEngine::CNNNetwork::size() const method
 * InferenceEngine::CNNNetwork::begin() const method
 * InferenceEngine::CNNNetwork::end() const method
 * InferenceEngine::CNNNetwork::AddExtension(const IShapeInferExtensionPtr& extension) method
 * InferenceEngine::ICNNNetwork::getPrecision() const noexcept method
 * InferenceEngine::ICNNNetwork::getName(char\* pName, size_t len) const noexcept method
 * InferenceEngine::ICNNNetwork::getData(const char\* dname) noexcept method
 * InferenceEngine::ICNNNetwork::addLayer(const CNNLayerPtr& layer) noexcept method
 * InferenceEngine::ICNNNetwork::getLayerByName(const char\* layerName, CNNLayerPtr& out, ResponseDesc\* resp) const noexcept method
 * InferenceEngine::ICNNNetwork::AddExtension(const IShapeInferExtensionPtr& extension, ResponseDesc\* resp) noexcept method
 * InferenceEngine::ICNNNetwork::getStats(ICNNNetworkStats\*\* stats, ResponseDesc\* resp) const noexcept method
 * InferenceEngine::ICNNNetworkStats class
 * InferenceEngine::NetworkNodeStats class
 * InferenceEngine::Data::getCreatorLayer() method
 * InferenceEngine::Data::getInputTo() method
 * InferenceEngine::LayerParams class

 **Layer API:**

 * InferenceEngine::CNNLayer class
 * InferenceEngine::WeightableLayer class
 * InferenceEngine::BatchNormalizationLayer class
 * InferenceEngine::BatchToSpaceLayer class
 * InferenceEngine::BinaryConvolutionLayer class
 * InferenceEngine::BroadcastLayer class
 * InferenceEngine::BucketizeLayer class
 * InferenceEngine::ClampLayer class
 * InferenceEngine::ConcatLayer class
 * InferenceEngine::ConvolutionLayer class
 * InferenceEngine::CropLayer class
 * InferenceEngine::DeconvolutionLayer class
 * InferenceEngine::DeformableConvolutionLayer class
 * InferenceEngine::DepthToSpaceLayer class
 * InferenceEngine::EltwiseLayer class
 * InferenceEngine::ExperimentalDetectronPriorGridGenerator class
 * InferenceEngine::ExperimentalDetectronPriorGridGeneratorLayer class
 * InferenceEngine::ExperimentalSparseWeightedReduceLayer class
 * InferenceEngine::FillLayer class
 * InferenceEngine::FullyConnectedLayer class
 * InferenceEngine::GRNLayer class
 * InferenceEngine::GRUCell class
 * InferenceEngine::GatherLayer class
 * InferenceEngine::GemmLayer class
 * InferenceEngine::LSTMCell class
 * InferenceEngine::MVNLayer class
 * InferenceEngine::MathLayer class
 * InferenceEngine::NonMaxSuppression class
 * InferenceEngine::NormLayer class
 * InferenceEngine::OneHotLayer class
 * InferenceEngine::PReLULayer class
 * InferenceEngine::PadLayer class
 * InferenceEngine::PoolingLayer class
 * InferenceEngine::PowerLayer class
 * InferenceEngine::QuantizeLayer class
 * InferenceEngine::RNNCell class
 * InferenceEngine::RNNCellBase class
 * InferenceEngine::RNNSequenceLayer class
 * InferenceEngine::RangeLayer class
 * InferenceEngine::ReLU6Layer class
 * InferenceEngine::ReLULayer class
 * InferenceEngine::ReduceLayer class
 * InferenceEngine::ReshapeLayer class
 * InferenceEngine::ReverseSequenceLayer class
 * InferenceEngine::ScaleShiftLayer class
 * InferenceEngine::ScatterLayer class
 * InferenceEngine::SelectLayer class
 * InferenceEngine::ShuffleChannelsLayer class
 * InferenceEngine::SoftMaxLayer class
 * InferenceEngine::SpaceToBatchLayer class
 * InferenceEngine::SpaceToDepthLayer class
 * InferenceEngine::SparseFillEmptyRowsLayer class
 * InferenceEngine::SparseSegmentReduceLayer class
 * InferenceEngine::SparseToDenseLayer class
 * InferenceEngine::SplitLayer class
 * InferenceEngine::StridedSliceLayer class
 * InferenceEngine::TensorIterator class
 * InferenceEngine::TileLayer class
 * InferenceEngine::TopKLayer class
 * InferenceEngine::UniqueLayer class

## 2020.4

### New API

 **CPU Plugin API:**

 * InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16 config key

 **Metrics and values for Query API:**

 * METRIC_KEY(OPTIMIZATION_CAPABILITIES)
	 * METRIC_VALUE(BF16)

### Deprecated API

 **MYRIAD Plugin API:**

 * VPU_CONFIG_KEY(IGNORE_IR_STATISTIC)

### Removed API

 **Inference Engine NN Builder API:**

 * InferenceEngine::Builder::EltwiseLayer
 * InferenceEngine::Builder::MemoryLayer
 * InferenceEngine::Builder::ROIPoolingLayer
 * InferenceEngine::Builder::DeconvolutionLayer
 * InferenceEngine::Builder::ReLULayer
 * InferenceEngine::Builder::TanHLayer
 * InferenceEngine::Builder::InputLayer
 * InferenceEngine::Builder::PoolingLayer
 * InferenceEngine::Builder::CropLayer
 * InferenceEngine::Builder::GRUSequenceLayer
 * InferenceEngine::Builder::NormLayer
 * InferenceEngine::Builder::LSTMSequenceLayer
 * InferenceEngine::Builder::ClampLayer
 * InferenceEngine::Builder::PSROIPoolingLayer
 * InferenceEngine::Builder::Layer
 * InferenceEngine::Builder::RNNSequenceLayer
 * InferenceEngine::Builder::ReorgYoloLayer
 * InferenceEngine::Builder::NormalizeLayer
 * InferenceEngine::Builder::PriorBoxClusteredLayer
 * InferenceEngine::Builder::MVNLayer
 * InferenceEngine::Builder::PermuteLayer
 * InferenceEngine::Builder::SimplerNMSLayer
 * InferenceEngine::Builder::ConstLayer
 * InferenceEngine::Builder::DeformableConvolutionLayer
 * InferenceEngine::Builder::FullyConnectedLayer
 * InferenceEngine::Builder::PriorBoxLayer
 * InferenceEngine::Builder::SoftMaxLayer
 * InferenceEngine::Builder::OutputLayer
 * InferenceEngine::Builder::TileLayer
 * InferenceEngine::Builder::SplitLayer
 * InferenceEngine::Builder::PReLULayer
 * InferenceEngine::Builder::RegionYoloLayer
 * InferenceEngine::Builder::ReshapeLayer
 * InferenceEngine::Builder::ConvolutionLayer
 * InferenceEngine::Builder::DetectionOutputLayer
 * InferenceEngine::Builder::ConcatLayer
 * InferenceEngine::Builder::ELULayer
 * InferenceEngine::Builder::GRNLayer
 * InferenceEngine::Builder::LRNLayer
 * InferenceEngine::Builder::ArgMaxLayer
 * InferenceEngine::Builder::ReLU6Layer
 * InferenceEngine::Builder::ScaleShiftLayer
 * InferenceEngine::Builder::ProposalLayer
 * InferenceEngine::Builder::SigmoidLayer
 * InferenceEngine::Builder::ResampleLayer
 * InferenceEngine::Builder::CTCGreedyDecoderLayer
 * InferenceEngine::Builder::BatchNormalizationLayer
 * InferenceEngine::Builder::LayerDecorator
 * InferenceEngine::Builder::PowerLayer
 * InferenceEngine::Builder::Network
 * InferenceEngine::Builder::PortInfo
 * InferenceEngine::Builder::Connection
 * InferenceEngine::Builder::PortData
 * InferenceEngine::Builder::Port
 * InferenceEngine::Builder::ILayer
 * InferenceEngine::Builder::INetworkIterator
 * InferenceEngine::Builder::INetwork
 * InferenceEngine::Builder::ILayer

## 2020.2

### New API

 **Extensibility API:**

 * InferenceEngine::IExtension::getImplTypes(const std::shared_ptr<ngraph::Node>& node) method
 * InferenceEngine::IExtension::getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) method

### Deprecated API

 **Extensibility API:**

 * InferenceEngine::ILayerImplFactory class
 * InferenceEngine::IShapeInferImpl class
 * InferenceEngine::IShapeInferImpl class
 * InferenceEngine::IShapeInferExtension class
 * InferenceEngine::IExtension::getFactoryFor(ILayerImplFactory\*& factory, const CNNLayer\* cnnLayer, ResponseDesc\* resp) noexcept method
 * InferenceEngine::IExtension::getPrimitiveTypes(char\*\*& types, unsigned int& size, ResponseDesc\* resp) noexcept method
 * InferenceEngine::ShapeInferImpl class
 * InferenceEngine::Extension::getFactoryFor(ILayerImplFactory\*& factory, const CNNLayer\* cnnLayer, ResponseDesc\* resp) noexcept method
 * InferenceEngine::Extension::getPrimitiveTypes(char\*\*& types, unsigned int& size, ResponseDesc\* resp) noexcept method

 **Network API:**

 * InferenceEngine::details::CNNNetworkIterator class
 * InferenceEngine::CNNNetwork::getPrecision() const method
 * InferenceEngine::CNNNetwork::getLayerByName(const char\* layerName) const method
 * InferenceEngine::CNNNetwork::size() const method
 * InferenceEngine::CNNNetwork::begin() const method
 * InferenceEngine::CNNNetwork::end() const method
 * InferenceEngine::CNNNetwork::AddExtension(const IShapeInferExtensionPtr& extension) method
 * InferenceEngine::ICNNNetwork::getPrecision() const noexcept method
 * InferenceEngine::ICNNNetwork::getName(char\* pName, size_t len) const noexcept method
 * InferenceEngine::ICNNNetwork::getData(const char\* dname) noexcept method
 * InferenceEngine::ICNNNetwork::addLayer(const CNNLayerPtr& layer) noexcept method
 * InferenceEngine::ICNNNetwork::getLayerByName(const char\* layerName, CNNLayerPtr& out, ResponseDesc\* resp) const noexcept method
 * InferenceEngine::ICNNNetwork::AddExtension(const IShapeInferExtensionPtr& extension, ResponseDesc\* resp) noexcept method
 * InferenceEngine::ICNNNetwork::getStats(ICNNNetworkStats\*\* stats, ResponseDesc\* resp) const noexcept method
 * InferenceEngine::ICNNNetworkStats class
 * InferenceEngine::NetworkNodeStats class
 * InferenceEngine::Data::getCreatorLayer() method
 * InferenceEngine::Data::getInputTo() method
 * InferenceEngine::LayerParams class

 **Layer API:**

 * InferenceEngine::CNNLayer class
 * InferenceEngine::WeightableLayer class
 * InferenceEngine::BatchNormalizationLayer class
 * InferenceEngine::BatchToSpaceLayer class
 * InferenceEngine::BinaryConvolutionLayer class
 * InferenceEngine::BroadcastLayer class
 * InferenceEngine::BucketizeLayer class
 * InferenceEngine::ClampLayer class
 * InferenceEngine::ConcatLayer class
 * InferenceEngine::ConvolutionLayer class
 * InferenceEngine::CropLayer class
 * InferenceEngine::DeconvolutionLayer class
 * InferenceEngine::DeformableConvolutionLayer class
 * InferenceEngine::DepthToSpaceLayer class
 * InferenceEngine::EltwiseLayer class
 * InferenceEngine::ExperimentalDetectronPriorGridGenerator class
 * InferenceEngine::ExperimentalDetectronPriorGridGeneratorLayer class
 * InferenceEngine::ExperimentalSparseWeightedReduceLayer class
 * InferenceEngine::FillLayer class
 * InferenceEngine::FullyConnectedLayer class
 * InferenceEngine::GRNLayer class
 * InferenceEngine::GRUCell class
 * InferenceEngine::GatherLayer class
 * InferenceEngine::GemmLayer class
 * InferenceEngine::LSTMCell class
 * InferenceEngine::MVNLayer class
 * InferenceEngine::MathLayer class
 * InferenceEngine::NonMaxSuppression class
 * InferenceEngine::NormLayer class
 * InferenceEngine::OneHotLayer class
 * InferenceEngine::PReLULayer class
 * InferenceEngine::PadLayer class
 * InferenceEngine::PoolingLayer class
 * InferenceEngine::PowerLayer class
 * InferenceEngine::QuantizeLayer class
 * InferenceEngine::RNNCell class
 * InferenceEngine::RNNCellBase class
 * InferenceEngine::RNNSequenceLayer class
 * InferenceEngine::RangeLayer class
 * InferenceEngine::ReLU6Layer class
 * InferenceEngine::ReLULayer class
 * InferenceEngine::ReduceLayer class
 * InferenceEngine::ReshapeLayer class
 * InferenceEngine::ReverseSequenceLayer class
 * InferenceEngine::ScaleShiftLayer class
 * InferenceEngine::ScatterLayer class
 * InferenceEngine::SelectLayer class
 * InferenceEngine::ShuffleChannelsLayer class
 * InferenceEngine::SoftMaxLayer class
 * InferenceEngine::SpaceToBatchLayer class
 * InferenceEngine::SpaceToDepthLayer class
 * InferenceEngine::SparseFillEmptyRowsLayer class
 * InferenceEngine::SparseSegmentReduceLayer class
 * InferenceEngine::SparseToDenseLayer class
 * InferenceEngine::SplitLayer class
 * InferenceEngine::StridedSliceLayer class
 * InferenceEngine::TensorIterator class
 * InferenceEngine::TileLayer class
 * InferenceEngine::TopKLayer class
 * InferenceEngine::UniqueLayer class

## 2020.1

### New API

 **Integration with ngraph API:**

 * InferenceEngine::CNNNetwork(const std::shared_ptr<ngraph::Function>& network) ctor from ngraph::Function
 * InferenceEngine::CNNNetwork::getFunction() const noexcept method
 * InferenceEngine::ICNNNetwork::getFunction() const noexcept method
 * InferenceEngine::Parameter(const std::shared_ptr<ngraph::Variant>& var) ctor
 * InferenceEngine::Parameter::asVariant() const method
 * InferenceEngine::Parameter::operator std::shared_ptr<ngraph::Variant>() const operator
 * InferenceEngine::Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) method
 * InferenceEngine::Core::ReadNetwork(const std::string& modelPath, const std::string& binPath = "") method
 * InferenceEngine::Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) method
 * InferenceEngine::Code::AddExtension(const IExtensionPtr& extension) method
 * InferenceEngine::IExtension::getOpSets() method


 **Offline compilation: import / export to std::stream:**

 * InferenceEngine::ExecutableNetwork::Export(std::ostream& networkModel) method
 * InferenceEngine::Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName = {}, const std::map<std::string, std::string>& config = {}) method
 * InferenceEngine::IExecutableNetwork::Export(std::ostream& networkModel, ResponseDesc \*resp) noexcept method


 **RemoteBlob accelerator memory sharing API:**

 * InferenceEngine::RemoteContext class
 * InferenceEngine::RemoteBlob class
 * InferenceEngine::Core::CreateContext(const std::string& deviceName, const ParamMap& params) method
 * InferenceEngine::Core::GetDefaultContext(const std::string& deviceName) method
 * InferenceEngine::Core::LoadNetwork(CNNNetwork network, RemoteContext::Ptr context, const std::map<std::string, std::string>& config = std::map<std::string, std::string>()) method


 **GNA firmware model image generation:**

  * GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION) config key
     * GNA_CONFIG_VALUE(GEN) value
     * GNA_CONFIG_VALUE(GEN_EXACT) value
     * GNA_CONFIG_VALUE(SSE) value
     * GNA_CONFIG_VALUE(SSE_EXACT) value
     * GNA_CONFIG_VALUE(AVX1) value
     * GNA_CONFIG_VALUE(AVX1_EXACT) value
     * GNA_CONFIG_VALUE(AVX2) value
     * GNA_CONFIG_VALUE(AVX2_EXACT) value

 **MemoryBlob mapping of memory to the user space:**

  * InferenceEngine::MemoryBlob::rwmap() noexcept method
  * InferenceEngine::MemoryBlob::rmap() noexcept method
  * InferenceEngine::MemoryBlob::wmap() noexcept method

 **Memory interoperability on acceleration devices. General classes and GPU helper functions**
  * InferenceEngine::RemoteBlob class
  * InferenceEngine::RemoteContext class
  * InferenceEngine::Core::CreateContext(const std::string& deviceName, const ParamMap& params) method
  * InferenceEngine::Core::GetDefaultContext(const std::string& deviceName) method
  * InferenceEngine::make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx) function
  * InferenceEngine::gpu::make_shared_blob_nv12(size_t height, size_t width, RemoteContext::Ptr ctx, VASurfaceID nv12_surf) function
  * InferenceEngine::gpu::make_shared_context(Core& core, std::string deviceName, VADisplay device) function
  * InferenceEngine::gpu::make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, VASurfaceID surface, uint32_t plane = 0) function
  * InferenceEngine::gpu::make_shared_blob_nv12(RemoteContext::Ptr ctx, cl::Image2D& nv12_image_plane_y, cl::Image2D& nv12_image_plane_uv) function
  * InferenceEngine::gpu::make_shared_context(Core& core, std::string deviceName, cl_context ctx) function
  * InferenceEngine::gpu::make_shared_blob(const TensorDesc& desc, ClContext::Ptr ctx) function
  * InferenceEngine::gpu::make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, cl::Buffer& buffer) function
  * InferenceEngine::gpu::make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, cl_mem buffer) function
  * InferenceEngine::gpu::make_shared_blob(const TensorDesc& desc, RemoteContext::Ptr ctx, cl::Image2D& image) function

### Deprecated API

 **Inference Engine NN Builder API:**

 * InferenceEngine::Builder::EltwiseLayer
 * InferenceEngine::Builder::MemoryLayer
 * InferenceEngine::Builder::ROIPoolingLayer
 * InferenceEngine::Builder::DeconvolutionLayer
 * InferenceEngine::Builder::ReLULayer
 * InferenceEngine::Builder::TanHLayer
 * InferenceEngine::Builder::InputLayer
 * InferenceEngine::Builder::PoolingLayer
 * InferenceEngine::Builder::CropLayer
 * InferenceEngine::Builder::GRUSequenceLayer
 * InferenceEngine::Builder::NormLayer
 * InferenceEngine::Builder::LSTMSequenceLayer
 * InferenceEngine::Builder::ClampLayer
 * InferenceEngine::Builder::PSROIPoolingLayer
 * InferenceEngine::Builder::Layer
 * InferenceEngine::Builder::RNNSequenceLayer
 * InferenceEngine::Builder::ReorgYoloLayer
 * InferenceEngine::Builder::NormalizeLayer
 * InferenceEngine::Builder::PriorBoxClusteredLayer
 * InferenceEngine::Builder::MVNLayer
 * InferenceEngine::Builder::PermuteLayer
 * InferenceEngine::Builder::SimplerNMSLayer
 * InferenceEngine::Builder::ConstLayer
 * InferenceEngine::Builder::DeformableConvolutionLayer
 * InferenceEngine::Builder::FullyConnectedLayer
 * InferenceEngine::Builder::PriorBoxLayer
 * InferenceEngine::Builder::SoftMaxLayer
 * InferenceEngine::Builder::OutputLayer
 * InferenceEngine::Builder::TileLayer
 * InferenceEngine::Builder::SplitLayer
 * InferenceEngine::Builder::PReLULayer
 * InferenceEngine::Builder::RegionYoloLayer
 * InferenceEngine::Builder::ReshapeLayer
 * InferenceEngine::Builder::ConvolutionLayer
 * InferenceEngine::Builder::DetectionOutputLayer
 * InferenceEngine::Builder::ConcatLayer
 * InferenceEngine::Builder::ELULayer
 * InferenceEngine::Builder::GRNLayer
 * InferenceEngine::Builder::LRNLayer
 * InferenceEngine::Builder::ArgMaxLayer
 * InferenceEngine::Builder::ReLU6Layer
 * InferenceEngine::Builder::ScaleShiftLayer
 * InferenceEngine::Builder::ProposalLayer
 * InferenceEngine::Builder::SigmoidLayer
 * InferenceEngine::Builder::ResampleLayer
 * InferenceEngine::Builder::CTCGreedyDecoderLayer
 * InferenceEngine::Builder::BatchNormalizationLayer
 * InferenceEngine::Builder::LayerDecorator
 * InferenceEngine::Builder::PowerLayer
 * InferenceEngine::Builder::Network
 * InferenceEngine::Builder::PortInfo
 * InferenceEngine::Builder::Connection
 * InferenceEngine::Builder::PortData
 * InferenceEngine::Builder::Port
 * InferenceEngine::Builder::ILayer
 * InferenceEngine::Builder::INetworkIterator
 * InferenceEngine::Builder::INetwork
 * InferenceEngine::Builder::ILayer

 **Plugin API:**

 * InferenceEngine::InferencePlugin C++ plugin wrapper class
 * InferenceEngine::IInferencePlugin plugin interface
 * InferenceEngine::PluginDispatcher class
 * InferenceEngine::InferenceEnginePluginPtr typedef
 * InferenceEngine::ICNNNetReader reader interface
 * InferenceEngine::CNNNetReader class

 **Blob API:**

  * Blob::element_size() const noexcept method
  * Blob::buffer() noexcept method
  * Blob::cbuffer() noexcept method
  * MemoryBlob::buffer() noexcept method
  * MemoryBlob::cbuffer() noexcept method


### Removed API

 Removed all [Inference Engine API which deprecated in 2019'R2](https://docs.openvinotoolkit.org/2019_R3/_docs_IE_DG_API_Changes.html#deprecated_api)

## 2019 R3

### New API

 **New supported layers:**

 * InferenceEngine::SparseFillEmptyRowsLayer new class
 * InferenceEngine::UniqueLayer new class
 * InferenceEngine::NonMaxSuppressionLayer new class
 * InferenceEngine::ScatterLayer new class

 **FPGA plugin streaming support:**

 * DLIA_METRIC_VALUE(INPUT_STREAMING) value to METRIC_KEY(OPTIMIZATION_CAPABILITIES)
 * DLIA_CONFIG_KEY(ENABLE_STREAMING) config key

### Removed API

 * InferenceEngine::EltwiseLayer::Select from InferenceEngine::EltwiseLayer::eOperation enumeration

## 2019 R2

### New API

 **Inference Engine Core API:**

 * Introduced InferenceEngine::Core high level class to manage devices

 **Query API extensions to InferenceEngine::ExecutableNetwork and InferenceEngine::IExecutableNetwork:**

 * InferenceEngine::ExecutableNetwork::SetConfig method
 * InferenceEngine::ExecutableNetwork::GetConfig method
 * InferenceEngine::ExecutableNetwork::GetMetric method
 * InferenceEngine::IExecutableNetwork::SetConfig method
 * InferenceEngine::IExecutableNetwork::GetConfig method
 * InferenceEngine::IExecutableNetwork::GetMetric method

 **Metrics and values for Query API:**

 * METRIC_KEY(AVAILABLE_DEVICES)
 * METRIC_KEY(SUPPORTED_METRICS)
 * METRIC_KEY(SUPPORTED_CONFIG_KEYS)
 * METRIC_KEY(FULL_DEVICE_NAME)
 * METRIC_KEY(OPTIMIZATION_CAPABILITIES)
	 * METRIC_VALUE(FP32)
	 * METRIC_VALUE(FP16)
	 * METRIC_VALUE(INT8)
	 * METRIC_VALUE(BIN)
	 * METRIC_VALUE(WINOGRAD)
	 * DLIA_METRIC_VALUE(FP11)
 * METRIC_KEY(RANGE_FOR_STREAMS)
 * METRIC_KEY(NUMBER_OF_WAITING_INFER_REQUESTS)
 * METRIC_KEY(NUMBER_OF_EXEC_INFER_REQUESTS)
 * METRIC_KEY(DEVICE_THERMAL)
 * METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)
 * EXEC_NETWORK_METRIC_KEY(NETWORK_NAME)
 * EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)

 **Common API:**

 * CLDNN_CONFIG_KEY(INT8_ENABLED) config key
 	* CONFIG_KEY(GPU_THROUGHPUT_AUTO)
	* CONFIG_KEY(GPU_THROUGHPUT_STREAMS)
 * DLIA_CONFIG_KEY(IO_TRANSFORMATIONS_NATIVE) config key
 * DLIA_CONFIG_KEY(DUMP_SUPPORTED_LAYERS_INFORMATION) config key
 * GNA_CONFIG_VALUE(SW_FP32) config value for GNA_CONFIG_KEY(DEVICE_MODE) key
 * MULTI_CONFIG_KEY(DEVICE_PRIORITIES) config key for `MULTI` device
 * InferenceEngine::CNNNetReader::ReadNetwork(const std::wstring &filepath) new method
 * InferenceEngine::CNNNetReader::ReadWeights(const std::wstring &filepath) new method
 * InferenceEngine::ExecutableNetwork::ExecutableNetwork(IExecutableNetwork::Ptr actual, InferenceEnginePluginPtr plg) constructor with additional `plg` parameter
 * InferenceEngine::InferRequest::InferRequest(IInferRequest::Ptr request, InferenceEnginePluginPtr plg) constructor with additional `plg` parameter
 * InferenceEngine::Data::setName method
 * InferenceEngine::QueryNetworkResult::supportedLayersMap
 * InferenceEngine::Precision::I64 extension to InferenceEngine::Precision::ePrecision enumeration

 **New supported primitives:**

 * InferenceEngine::Builder::DeformableConvolutionLayer new class
 * InferenceEngine::DeformableConvolutionLayer new class
 * InferenceEngine::EltwiseLayer::Logical_NOT, InferenceEngine::EltwiseLayer::Mean, InferenceEngine::EltwiseLayer::Select extensions to InferenceEngine::EltwiseLayer::eOperation enumeration
 * InferenceEngine::OneHotLayer new class
 * InferenceEngine::SelectLayer new class
 * InferenceEngine::BroadcastLayer new class
 * InferenceEngine::MathLayer new class
 * InferenceEngine::ReduceLayer new class
 * InferenceEngine::TopKLayer new class

 **Extensions to Blob creation API:**

 * InferenceEngine::Blob::is method
 * InferenceEngine::Blob::is const method
 * InferenceEngine::Blob::as method
 * InferenceEngine::Blob::as const method
 * InferenceEngine::Blob::getAllocator abstract method
 * InferenceEngine::Blob::getHandle abstract method
 * InferenceEngine::MemoryBlob class
 * InferenceEngine::ColorFormat enumeration
 * InferenceEngine::PreProcessInfo::setColorFormat method
 * InferenceEngine::PreProcessInfo::getColorFormat method
 * InferenceEngine::CompoundBlob class to work with blobs consisting of several planes
 * InferenceEngine::NV12Blob class representing NV12 blob with two planes

### Deprecated API

The methods listed below are deprecated and will be removed in 2019 R4 release:

 **Common API:**

 * InferenceEngine::InputInfo::getInputPrecision method
 * InferenceEngine::InputInfo::setInputPrecision method
 * InferenceEngine::InputInfo::getDims method
 * InferenceEngine::CNNLayer::GetParamsAsBool method
 * InferenceEngine::CNNNetwork::CNNNetwork(ICNNNetwork* actual) constructor
 * InferenceEngine::CNNNetwork::setTargetDevice method
 * HETERO_CONFIG_KEY(DUMP_DLA_MESSAGES) config key
 * InferenceEngine::ILayerImplFactory::getShapes method
 * InferenceEngine::IShapeInferImpl::inferShapes(const std::vector<SizeVector>&, const std::map<std::string, std::string>& , const std::map<std::string, Blob::Ptr>&, std::vector<SizeVector>&, ResponseDesc\*) method
 * InferenceEngine::Data::setBatchSize method
 * InferenceEngine::QueryNetworkResult::supportedLayers field
 * InferenceEngine::ICNNNetwork::setBatchSize(const size_t size) method
 * InferenceEngine::Blob::Resize method
 * InferenceEngine::Blob::Reshape method
 * InferenceEngine::TBlob::set method

 **InferenceEngine::IInferencePlugin and InferenceEngine:InferencePlugin obsolete methods:**

 * InferenceEngine::InferencePlugin::LoadNetwork(ICNNNetwork &network) method
 * InferenceEngine::InferencePlugin::Infer method
 * InferenceEngine::InferencePlugin::GetPerformanceCounts method
 * InferenceEngine::InferencePlugin::QueryNetwork(const ICNNNetwork &network, QueryNetworkResult &res) const method
 * InferenceEngine::IInferencePlugin::LoadNetwork(ICNNNetwork &network, ResponseDesc \*resp) method
 * InferenceEngine::IInferencePlugin::Infer(const Blob &input, Blob &result, ResponseDesc \*resp) method
 * InferenceEngine::IInferencePlugin::Infer(const BlobMap &input, BlobMap &result, ResponseDesc \*resp) method
 * InferenceEngine::IInferencePlugin::GetPerformanceCounts method
 * InferenceEngine::IInferencePlugin::QueryNetwork(const ICNNNetwork& network, QueryNetworkResult& res) const method


 **Fields in InferenceEngine::Data class are replaced with appropriate methods:**

 * InferenceEngine::Data::precision field
 * InferenceEngine::Data::layout field
 * InferenceEngine::Data::dims field
 * InferenceEngine::Data::creatorLayer field
 * InferenceEngine::Data::name field
 * InferenceEngine::Data::inputTo field
 * InferenceEngine::Data::userObject field

 **Heterogeneous plugin:**

 * InferenceEngine::IHeteroDeviceLoader class
 * InferenceEngine::IHeteroInferencePlugin class
 * InferenceEngine::HeteroPluginPtr class
 * operator InferenceEngine::InferencePlugin::HeteroPluginPtr operator

 **Blob creation API with dimensions in reverse order:**

 * InferenceEngine::Blob::Blob(Precision p) constructor
 * InferenceEngine::Blob::Blob(Precision p, Layout l) constructor
 * InferenceEngine::Blob::Blob(Precision p, const SizeVector &dims) constructor
 * InferenceEngine::Blob::Blob(Precision p, Layout l, const SizeVector &dims) constructor
 * InferenceEngine::TBlob::TBlob(Precision p, Layout l) constructor
 * InferenceEngine::TBlob::TBlob(Precision p, Layout l, const SizeVector& dims) constructor
 * InferenceEngine::TBlob::TBlob(Precision p, Layout l, const SizeVector& dims, T* ptr, size_t data_size) constructor
 * InferenceEngine::TBlob::TBlob(Precision p, Layout l, const SizeVector &dims, std::shared_ptr<IAllocator> alloc) constructor
 * InferenceEngine::Blob::type() method
 * InferenceEngine::Blob::precision() method
 * InferenceEngine::Blob::layout() method
 * InferenceEngine::Blob::dims() method
 * InferenceEngine::make_shared_blob(Precision p, Layout l, const SizeVector &dims) function
 * InferenceEngine::make_shared_blob(Precision p, const SizeVector &dims) function
 * InferenceEngine::make_shared_blob(Precision p, Layout l, const TArg &arg) function
 * InferenceEngine::make_shared_blob(Precision p, const TArg &arg) function
 * InferenceEngine::make_shared_blob(TBlob<TypeTo> &&arg) function
 * InferenceEngine::make_shared_blob(Precision p, Layout l) function
 * InferenceEngine::make_shared_blob(Precision p, Layout l, SizeVector dims, const std::vector<TypeTo> &arg) function
 * InferenceEngine::make_shared_blob(Precision p, Layout l, const std::vector<TypeTo> &arg) function
 * InferenceEngine::make_shared_blob(Precision p, const std::vector<TypeTo> &arg) function
 * InferenceEngine::make_shared_blob(Precision p, Layout l, const SizeVector &dims, TypeTo * ptr, size_t size) function
 * InferenceEngine::make_shared_blob(Precision p, const SizeVector &dims, TypeTo * ptr, size_t size) function
 * InferenceEngine::I_N variable
 * InferenceEngine::I_C variable
 * InferenceEngine::I_H variable
 * InferenceEngine::I_W variable
 * InferenceEngine::LayoutOffsetCounter class
 * InferenceEngine::ConvertLayout function

 **API working with device enumeration:**

 * InferenceEngine::TargetDevice enumeration
 * InferenceEngine::TargetDeviceInfo class
 * InferenceEngine::getDeviceName function
 * InferenceEngine::FindPluginRequest class
 * InferenceEngine::FindPluginResponse class
 * InferenceEngine::findPlugin(const FindPluginRequest &req, FindPluginResponse &result, ResponseDesc *resp) function
 * InferenceEngine::ICNNNetwork::setTargetDevice method
 * InferenceEngine::ICNNNetwork::getTargetDevice method
 * InferenceEngine::PluginDispatcher::getPluginByDevice method
 * InferenceEngine::PluginDispatcher::getSuitablePlugin method
