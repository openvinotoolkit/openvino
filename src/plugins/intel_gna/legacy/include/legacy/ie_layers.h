// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for internal Layers structure to describe layers information
 *
 * @file ie_layers.h
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <legacy/ie_layers_property.hpp>
#include <limits>
#include <map>
#include <memory>
#include <ngraph/node.hpp>
#include <string>
#include <vector>

#include "ie_blob.h"
#include "ie_common.h"
#include "ie_data.h"

#ifdef _MSC_VER
#    define _IE_SUPPRESS_DEPRECATED_START_MSVC IE_SUPPRESS_DEPRECATED_START
#    define _IE_SUPPRESS_DEPRECATED_END_MSVC   IE_SUPPRESS_DEPRECATED_END
#else
#    define _IE_SUPPRESS_DEPRECATED_START_MSVC
#    define _IE_SUPPRESS_DEPRECATED_END_MSVC
#endif

namespace InferenceEngine {

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 */
struct LayerParams {
    /**
     * @brief Layer name
     */
    std::string name;

    /**
     * @brief Layer type
     */
    std::string type;

    /**
     * @brief Layer precision
     */
    Precision precision;

    /**
     * @brief A default constructor.
     */
    LayerParams();

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief A copy constructor.
     * @param other An object to copy.
     */
    LayerParams(const LayerParams& other);

    /**
     * @brief A copy assignment operator
     * @param other An object to copy
     * @return A value
     */
    LayerParams& operator=(const LayerParams& other);

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A constructor with parameters.
     * @param name A layer name.
     * @param type A layer type.
     * @param precision A layer precision.
     */
    LayerParams(const std::string& name, const std::string& type, Precision precision);
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This is a base abstraction Layer - all DNN Layers inherit from this class
 */
class CNNLayer {
protected:
    std::shared_ptr<ngraph::Node> node;

public:
    /**
     * @brief A shared pointer to CNNLayer
     */
    IE_SUPPRESS_DEPRECATED_START
    using Ptr = std::shared_ptr<CNNLayer>;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Layer name
     */
    std::string name;

    /**
     * @brief Layer type
     */
    std::string type;

    /**
     * @brief Layer base operating precision
     */
    Precision precision;

    /**
     * @brief A vector of pointers to the output data elements of this layer in the di-graph (order matters)
     */
    std::vector<DataPtr> outData;

    /**
     * @brief A vector of weak pointers to the input data elements of this layer in the di-graph (order matters)
     */
    std::vector<DataWeakPtr> insData;

    /**
     * @brief If suggested to fuse - a pointer to the layer which needs to be fused with this layer
     */
    _IE_SUPPRESS_DEPRECATED_START_MSVC
    Ptr _fusedWith;
    _IE_SUPPRESS_DEPRECATED_END_MSVC

    /**
     * @brief Convenience user values to store in this object as extra data
     */
    UserValue userValue;

    /**
     * @brief Layer affinity set by user.
     */
    std::string affinity;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief A constructor. Creates a new CNNLayer instance and initializes layer parameters with the given values.
     *
     * @param prms Basic common parsing parameters
     */
    explicit CNNLayer(const LayerParams& prms);

    /**
     * @brief Returns the original nGraph op
     * @return A smart pointer to nGraph op
     */
    std::shared_ptr<ngraph::Node> getNode() const {
        return node;
    }

    /**
     * @brief A copy constructor
     * @param other An object to copy
     */
    CNNLayer(const CNNLayer& other);

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A virtual destructor
     */
    virtual ~CNNLayer();

    /**
     * @brief Sets a layer to be fused with
     *
     * @param layer Reference to the layer to be fused with
     */
    _IE_SUPPRESS_DEPRECATED_START_MSVC
    void fuse(Ptr& layer) {
        _fusedWith = layer;
    }
    _IE_SUPPRESS_DEPRECATED_END_MSVC

    /**
     * @brief Returns the first element of the input data for this layer
     *
     * @return A smart pointer to the input data element
     */
    virtual const DataPtr input() const;

    /**
     * @brief Checks if the input data and layer data are legitimate
     */
    void parseParams();

    /**
     * @brief Parse string with float in accordance with IE rules
     *
     * @param str input string with float value
     * @return float value if parsing was successful
     * @throws Exception in case of parsing error
     */
    static float ie_parse_float(const std::string& str);

    /**
     * @brief serialize float with c_locale formating
     * used for default values serializing
     */
    static std::string ie_serialize_float(float value);

    /**
     * @brief Gets float value for the given parameter
     *
     * @param param name of the parameter to find
     * @param def default value of the parameter if not found
     * @return float value
     */
    float GetParamAsFloat(const char* param, float def) const;

    /**
     * @brief Returns a float value for the given layer parameter
     *
     * @param param Name of the layer parameter
     * @return A float value for the specified parameter
     */
    float GetParamAsFloat(const char* param) const;

    /**
     * @brief Returns a vector of float values for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char* param, std::vector<float> def) const;

    /**
     * @brief Returns a vector of float values for the given parameter
     *
     * @param param Name of the layer parameter
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char* param) const;

    /**
     * @brief Returns an integer value for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char* param, int def) const;

    /**
     * @brief Returns an integer value for the given parameter
     *
     * @param param Name of the layer parameter
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char* param) const;

    /**
     * @brief Returns a vector of int values for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of int values
     */
    std::vector<int> GetParamAsInts(const char* param, std::vector<int> def) const;

    /**
     * @brief Returns a vector of int values for the given parameter
     *
     * @param param Name of the layer parameter
     * @return vector of int values
     */
    std::vector<int> GetParamAsInts(const char* param) const;

    /**
     * @brief Returns an unsigned integer value for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int GetParamAsUInt(const char* param, unsigned int def) const;

    /**
     * @brief Returns an unsigned integer value for the given parameter
     *
     * @param param Name of the layer parameter
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int GetParamAsUInt(const char* param) const;

    /**
     * @brief Returns an size_t value for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An size_t value for the specified parameter
     */
    size_t GetParamAsSizeT(const char* param, size_t def) const;

    /**
     * @brief Returns an size_t value for the given parameter
     *
     * @param param Name of the layer parameter
     * @return An size_t value for the specified parameter
     */
    size_t GetParamAsSizeT(const char* param) const;

    /**
     * @brief Returns a vector of unsigned int values for the given parameter or returns the default value
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> GetParamAsUInts(const char* param, std::vector<unsigned int> def) const;

    /**
     * @brief Returns a vector of unsigned int values for the given parameter
     *
     * @param param Name of the layer parameter
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> GetParamAsUInts(const char* param) const;

    /**
     * @brief Returns a boolean value for the given parameter.
     *
     * The valid values are (true, false, 1, 0).
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return A bool value for the specified parameter
     */
    bool GetParamAsBool(const char* param, bool def) const;

    /**
     * @brief Returns a boolean value for the given parameter
     *
     * @param param Name of the layer parameter
     * @return A bool value for the specified parameter
     */
    bool GetParamAsBool(const char* param) const;

    /**
     * @brief Returns a string value for the given parameter or returns the default one
     *
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return A string value
     */
    std::string GetParamAsString(const char* param, const char* def) const;

    /**
     * @brief Checks the param presence in the layer
     *
     * @param param Name of the layer parameter
     * @return a bool depending param presence
     */
    bool CheckParamPresence(const char* param) const;

    /**
     * @brief Returns a string value for the given parameter.
     *
     * Throws exception if parameter was not found.
     * @param param Name of the layer parameter
     * @return A string value
     */
    std::string GetParamAsString(const char* param) const;

    /**
     * @brief Returns a string containing an integer if parameters value was
     * "true" or "false"
     *
     * @param param Name of the layer parameter
     * @return A string containing an integer or the parameter as string
     */
    std::string getBoolStrParamAsIntStr(const char* param) const;

    /**
     * @brief Gets the parameter as a std::vector<std::string>
     * @param param  The parameter name
     * @param def The default values if case of parameter is not found
     * @return The parameter as strings.
     */
    std::vector<std::string> GetParamAsStrings(const char* param, std::vector<std::string> def) const;

    /**
     * @brief Map of pairs: (parameter name, parameter value)
     */
    std::map<std::string, std::string> params;

    /**
     * @brief Map of pairs: (name, weights/biases blob)
     */
    std::map<std::string, Blob::Ptr> blobs;
};

IE_SUPPRESS_DEPRECATED_START

/**
 * @brief Alias for CNNLayer object
 */
using GenericLayer = class CNNLayer;

/**
 * @brief A smart pointer to the CNNLayer
 */
using CNNLayerPtr = std::shared_ptr<CNNLayer>;
/**
 * @brief A smart weak pointer to the CNNLayer
 */
using CNNLayerWeakPtr = std::weak_ptr<CNNLayer>;

CNNLayerWeakPtr& getCreatorLayer(const DataPtr& data);

std::map<std::string, CNNLayerPtr>& getInputTo(const DataPtr& data);
std::map<std::string, CNNLayerPtr>& getInputTo(Data* data);
IE_SUPPRESS_DEPRECATED_END

_IE_SUPPRESS_DEPRECATED_START_MSVC

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a layer with Weights and/or Biases (e.g. Convolution/Fully Connected, etc.)
 */
class WeightableLayer : public CNNLayer {
public:
    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief A default constructor. Constructs a WeightableLayer instance and initiates layer parameters with the given
     * values
     *
     * @param prms Initial layer parameters
     */
    explicit WeightableLayer(const LayerParams& prms);

    /**
     * @brief Constructs a WeightableLayer instance and initiates layer parameters with the given values
     */
    using CNNLayer::CNNLayer;

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief A pointer to a weights blob
     */
    Blob::Ptr _weights;
    /**
     * @brief A pointer to a biases blob
     */
    Blob::Ptr _biases;

    ~WeightableLayer() override;
};

/**
 * @brief convinenent way to declare property with backward compatibility to 2D members
 */
#define DEFINE_PROP(prop_name)                          \
    PropertyVector<unsigned int> prop_name;             \
    unsigned int& prop_name##_x = prop_name.at(X_AXIS); \
    unsigned int& prop_name##_y = prop_name.at(Y_AXIS)

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard 3D Convolution Layer
 */
class ConvolutionLayer : public WeightableLayer {
public:
    /**
     * @brief A convolution kernel array [X, Y, Z, ...]
     */
    DEFINE_PROP(_kernel);
    /**
     * @brief A convolution paddings begin array [X, Y, Z, ...]
     */
    DEFINE_PROP(_padding);
    /**
     * @brief A convolution paddings end array [X, Y, Z, ...]
     */
    DEFINE_PROP(_pads_end);
    /**
     * @brief A convolution strides array [X, Y, Z, ...]
     */
    DEFINE_PROP(_stride);
    /**
     * @brief A convolution dilations array [X, Y, Z, ...]
     */
    DEFINE_PROP(_dilation);
    /**
     * @brief A number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth = 0u;
    /**
     * @brief Number of groups
     */
    unsigned int _group = 1u;
    /**
     * @brief Auto padding type
     */
    std::string _auto_pad;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Creates a new ConvolutionLayer instance.
     */
    explicit ConvolutionLayer(const LayerParams& p)
        : WeightableLayer(p),
          _kernel(2, 0u),
          _padding(2, 0u),
          _stride(2, 1u),
          _dilation(2, 1u) {}

    /**
     * @brief assignment operator
     */
    ConvolutionLayer& operator=(const ConvolutionLayer& that) {
        if (&that != this) {
            WeightableLayer::operator=(that);
            _kernel = that._kernel;
            _padding = that._padding;
            _pads_end = that._pads_end;
            _stride = that._stride;
            _dilation = that._dilation;
            _out_depth = that._out_depth;
            _group = that._group;
        }
        return *this;
    }

    /**
     * @brief copy constructor
     */
    ConvolutionLayer(const ConvolutionLayer& that) : WeightableLayer(that) {
        operator=(that);
    }
    /**
     * @brief move constructor
     */
    ConvolutionLayer(ConvolutionLayer&&) = default;

    IE_SUPPRESS_DEPRECATED_END

    ~ConvolutionLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard deconvolution layer
 */
class DeconvolutionLayer : public ConvolutionLayer {
public:
    IE_SUPPRESS_DEPRECATED_START
    using ConvolutionLayer::ConvolutionLayer;
    using ConvolutionLayer::operator=;
    IE_SUPPRESS_DEPRECATED_END

    ~DeconvolutionLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard deformable convolution layer
 */
class DeformableConvolutionLayer : public ConvolutionLayer {
public:
    IE_SUPPRESS_DEPRECATED_START
    using ConvolutionLayer::ConvolutionLayer;
    using ConvolutionLayer::operator=;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Number of deformable groups
     */
    unsigned int _deformable_group = 1u;

    ~DeformableConvolutionLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard pooling layer
 */
class PoolingLayer : public CNNLayer {
public:
    /**
     * @brief Pooling kernel array [X, Y, Z, ...]
     */
    DEFINE_PROP(_kernel);
    /**
     * @brief Pooling paddings begin array [X, Y, Z, ...]
     */
    DEFINE_PROP(_padding);
    /**
     * @brief Pooling paddings end array [X, Y, Z, ...]
     */
    PropertyVector<unsigned int> _pads_end;
    /**
     * @brief Pooling strides array [X, Y, Z, ...]
     */
    DEFINE_PROP(_stride);

    /**
     * @enum PoolType
     * @brief Defines available pooling types
     */
    enum PoolType { MAX = 1, AVG = 2, STOCH = 3, ROI = 4, SPACIAL_PYRAMID = 5 };

    /**
     * @brief A pooling type
     */
    PoolType _type = MAX;

    /**
     * @brief A flag that indicates if padding is excluded or not
     */
    bool _exclude_pad = false;
    /**
     * @brief Auto padding type
     */
    std::string _auto_pad;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Creates a new PoolingLayer instance.
     */
    explicit PoolingLayer(const LayerParams& p) : CNNLayer(p), _kernel(2, 0u), _padding(2, 0u), _stride(2, 0u) {}

    /**
     * @brief assignment operator
     */
    PoolingLayer& operator=(const PoolingLayer& that) {
        if (&that != this) {
            CNNLayer::operator=(that);
            _kernel = that._kernel;
            _padding = that._padding;
            _pads_end = that._pads_end;
            _stride = that._stride;
            _type = that._type;
            _exclude_pad = that._exclude_pad;
        }
        return *this;
    }
    /**
     * @brief copy constructor
     */
    PoolingLayer(const PoolingLayer& that) : CNNLayer(that) {
        operator=(that);
    }

    /**
     * @brief move constructor
     */
    PoolingLayer(PoolingLayer&&) = default;

    IE_SUPPRESS_DEPRECATED_END

    ~PoolingLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard binary convolution layer
 */
class BinaryConvolutionLayer : public WeightableLayer {
public:
    /**
     * @enum eBinaryConvolutionMode
     * @brief Defines possible modes of binary convolution operation
     */
    enum eBinaryConvolutionMode { xnor_popcount = 0 };

    /**
     * @brief Mode of binary convolution operation
     */
    eBinaryConvolutionMode _mode = xnor_popcount;

    /**
     * @brief A number of input feature maps (size) generating the 3'rd input dimension
     */
    unsigned int _in_depth = 0u;

    /**
     * @brief A pad value which is used to fill pad area
     */
    float _pad_value = 0.0f;

    /**
     * @brief A convolution kernel array [X, Y, Z, ...]
     */
    DEFINE_PROP(_kernel);
    /**
     * @brief A convolution paddings begin array [X, Y, Z, ...]
     */
    DEFINE_PROP(_padding);
    /**
     * @brief A convolution paddings end array [X, Y, Z, ...]
     */
    PropertyVector<unsigned int> _pads_end;
    /**
     * @brief A convolution strides array [X, Y, Z, ...]
     */
    DEFINE_PROP(_stride);
    /**
     * @brief A convolution dilations array [X, Y, Z, ...]
     */
    DEFINE_PROP(_dilation);
    /**
     * @brief A number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth = 0u;
    /**
     * @brief Number of groups
     */
    unsigned int _group = 1u;
    /**
     * @brief Auto padding type
     */
    std::string _auto_pad;

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Creates a new BinaryConvolutionLayer instance.
     */
    explicit BinaryConvolutionLayer(const LayerParams& p)
        : WeightableLayer(p),
          _kernel(2, 0u),
          _padding(2, 0u),
          _stride(2, 1u),
          _dilation(2, 1u) {}

    /**
     * @brief assignment operator
     */
    BinaryConvolutionLayer& operator=(const BinaryConvolutionLayer& that) {
        if (&that != this) {
            WeightableLayer::operator=(that);
            _kernel = that._kernel;
            _padding = that._padding;
            _pads_end = that._pads_end;
            _stride = that._stride;
            _dilation = that._dilation;
            _out_depth = that._out_depth;
            _group = that._group;
            _mode = that._mode;
            _in_depth = that._in_depth;
            _pad_value = that._pad_value;
        }
        return *this;
    }
    /**
     * @brief copy constructor
     */
    BinaryConvolutionLayer(const BinaryConvolutionLayer& that) : WeightableLayer(that) {
        operator=(that);
    }
    /**
     * @brief move constructor
     */
    BinaryConvolutionLayer(BinaryConvolutionLayer&&) = default;

    IE_SUPPRESS_DEPRECATED_END

    ~BinaryConvolutionLayer() override;
};

#undef DEFINE_PROP

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a fully connected layer
 */
class FullyConnectedLayer : public WeightableLayer {
public:
    /**
     * @brief A size of output
     */
    unsigned int _out_num = 0;

    /**
     * @brief Creates a new FullyConnectedLayer instance and initializes layer parameters with the given values.
     */
    IE_SUPPRESS_DEPRECATED_START
    using WeightableLayer::WeightableLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~FullyConnectedLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents concatenation layer
 *
 * Takes as input several data elements and merges them to one using the supplied axis
 */
class ConcatLayer : public CNNLayer {
public:
    /**
     * @brief An axis on which concatenation operation is performed
     */
    unsigned int _axis = 1;

    /**
     * @brief Creates a new ConcatLayer instance and initializes layer parameters with the given values.
     *
     * If batch is used, then batch needs to be specified as an input dimension also
     * In current implementation 1 means channels, 0 - batch
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ConcatLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a layer that evenly splits the input into the supplied outputs
 */
class SplitLayer : public CNNLayer {
public:
    /**
     * @brief An axis on which split operation is performed
     */
    unsigned int _axis = 1;

    /**
     * @brief Creates a new SplitLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SplitLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Linear Response Normalization (LRN) Layer
 */
class NormLayer : public CNNLayer {
public:
    /**
     * @brief Response size
     */
    unsigned int _size = 0;
    /**
     * @brief K
     */
    unsigned int _k = 1;
    /**
     * @brief Alpha coefficient
     */
    float _alpha = 0;
    /**
     * @brief Beta coefficient
     */
    float _beta = 0;
    /**
     * @brief Flag to specify normalization across feature maps (true) or across channels
     */
    bool _isAcrossMaps = false;

    /**
     * @brief Creates a new NormLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~NormLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents standard softmax Layer
 */
class SoftMaxLayer : public CNNLayer {
public:
    /**
     * @brief Axis number for a softmax operation
     */
    int axis = 1;
    /**
     * @brief Creates a new SoftMaxLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SoftMaxLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents standard GRN Layer
 */
class GRNLayer : public CNNLayer {
public:
    /**
     * @brief A default constructor. Creates a new GRNLayer instance and initializes layer parameters with the given
     * values.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Bias for squares sum
     */
    float bias = 0.f;

    ~GRNLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents standard MVN Layer
 */
class MVNLayer : public CNNLayer {
public:
    /**
     * @brief A default constructor. Creates a new MVNLayer instance and initializes layer parameters with the given
     * values.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Indicate that mean value is calculated across channels
     */
    int across_channels = 0;

    /**
     * @brief Indicate that the result needs to be normalized
     */
    int normalize = 1;

    ~MVNLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Rectified Linear activation layer
 */
class ReLULayer : public CNNLayer {
public:
    /**
     * @brief Negative slope is used to takle negative inputs instead of setting them to 0
     */
    float negative_slope = 0.0f;

    /**
     * @brief Creates a new ReLULayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ReLULayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Clamp activation layer
 *
 * Clamps all tensor elements into the range [min_value, max_value]
 */
class ClampLayer : public CNNLayer {
public:
    /**
     * @brief A minimum value
     */
    float min_value = 0.0f;

    /**
     * @brief A maximum value
     */
    float max_value = 1.0f;
    /**
     * @brief Creates a new ClampLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ClampLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a ReLU6 activation layer
 *
 * Clamps all tensor elements into the range [0, 6.0]
 */
class ReLU6Layer : public ClampLayer {
public:
    IE_SUPPRESS_DEPRECATED_START
    /**
     * @brief A constructor with common layer parameters
     * @param prms The common layer parameters
     */
    explicit ReLU6Layer(const LayerParams& prms) : ClampLayer(prms) {
        max_value = 6.0f;
    }
    IE_SUPPRESS_DEPRECATED_END

    ~ReLU6Layer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents an element wise operation layer
 */
class EltwiseLayer : public CNNLayer {
public:
    /**
     * @enum eOperation
     * @brief Defines possible operations that can be used
     */
    enum eOperation {
        Sum = 0,
        Prod,
        Max,
        Sub,
        Min,
        Div,
        Squared_diff,
        Floor_mod,
        Pow,
        Equal,
        Not_equal,
        Less,
        Less_equal,
        Greater,
        Greater_equal,
        Logical_AND,
        Logical_OR,
        Logical_XOR,
        Logical_NOT,
        Mean,
        Abs,
    };

    /**
     * @brief A type of the operation to use
     */
    eOperation _operation = Sum;

    /**
     * @brief A vector of coefficients to scale the operands
     */
    std::vector<float> coeff;

    /**
     * @brief Creates a new EltwiseLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~EltwiseLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard crop layer
 */
class CropLayer : public CNNLayer {
public:
    /**
     * @brief A vector of dimensions for cropping
     */
    std::vector<int> axis;
    /**
     * @brief A vector of dimensions to be preserved
     */
    std::vector<int> dim;
    /**
     * @brief A vector of offsets for each dimension
     */
    std::vector<int> offset;

    /**
     * @brief Creates a new CropLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~CropLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard reshape layer
 */
class ReshapeLayer : public CNNLayer {
public:
    /**
     * @brief A vector of sizes of the shape
     */
    std::vector<int> shape;
    /**
     * @brief A number of axis to be taken for a reshape
     */
    int axis = 0;
    /**
     * @brief A number of first axises to be taken for a reshape
     */
    int num_axes = -1;

    /**
     * @brief Creates a new ReshapeLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ReshapeLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Tile Layer
 */
class TileLayer : public CNNLayer {
public:
    /**
     * @brief An index of the axis to tile
     */
    int axis = -1;
    /**
     * @brief A number of copies to be made
     */
    int tiles = -1;

    /**
     * @brief Creates a new TileLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~TileLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Layer which performs Scale and Shift
 */
class ScaleShiftLayer : public WeightableLayer {
public:
    /**
     * @brief A flag that indicates if the same value is used for all the features. If false, the value is used pixel
     * wise
     */
    unsigned int _broadcast = 0;

    /**
     * @brief Creates a new ScaleShiftLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using WeightableLayer::WeightableLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ScaleShiftLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents TensorIterator layer
 */
class TensorIterator : public CNNLayer {
public:
    struct PortMap {
        // Data map rule
        int from; /**< Index of exteral data from ins/outs fields of CNNLayer */
        int to;   /**< Index of internal data in iterator body */

        // Iteration rule
        int axis;      /**< Axis to iterate throught */
        int stride;    /**< Stride to iterate throught */
        int start;     /**< Start index of iteration range */
        int end;       /**< Last index of iteration range  */
        int part_size; /**< Part size which will be transfered to body subnetwork */
    };

    /**
     * @brief Describes a tensor iterator body
     */
    struct Body {
        std::vector<DataPtr> inputs;   //!< Inputs data
        std::vector<DataPtr> outputs;  //!< Outputs data
    };

    std::vector<PortMap> input_port_map;   //!< Input ports map
    std::vector<PortMap> output_port_map;  //!< Output ports map
    std::vector<PortMap> back_edges;       //!< Back edges map

    Body body;  //!< A Tensor Iterator body

    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~TensorIterator() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief Base class for recurrent cell layers
 */
class RNNCellBase : public WeightableLayer {
public:
    IE_SUPPRESS_DEPRECATED_START
    using WeightableLayer::WeightableLayer;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Direct type of recurrent cell (including subtypes)
     * Description of particular cell semantics is in LSTMCell, GRUCell, RNNCell.
     */
    enum CellType {
        LSTM,    /**< Original LSTM cell */
        GRU,     /**< Original GRU cell */
        RNN,     /**< Original RNN cell */
        GRU_LBR, /**< GRU cell modification. "Linear before reset" */
    };

    /** @copybrief CellType */
    CellType cellType = LSTM;

    /**
     * @brief Size of hidden state data
     *
     * In case of batch output state tensor will have shape [N, hidden_size]
     */
    int hidden_size = 0;

    /**
     * @brief Clip data into range [-clip, clip] on input of activations
     *
     * clip==0.0f means no clipping
     */
    float clip = 0.0f;
    /**
     * @brief Activations used inside recurrent cell
     *
     * Valid values: sigmoid, tanh, relu
     */
    std::vector<std::string> activations;

    /**
     * @brief Alpha parameters of activations
     *
     * Respective to activation list.
     */
    std::vector<float> activation_alpha;

    /**
     * @brief Beta parameters of activations
     *
     * Respective to activation list.
     */
    std::vector<float> activation_beta;

    ~RNNCellBase() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief LSTM Cell layer
 *
 * G - number of gates (=4)
 * N - batch size
 * S - state size (=hidden_size)
 *
 * Inputs:
 *   [N,D] Xt - input data
 *   [N,S] Ht-1 - initial hidden state
 *   [N,S] Ct-1 - initial cell state
 *
 * Outputs:
 *   [N,S] Ht - out hidden state
 *   [N,S] Ct - out cell state
 *
 * Weights:
 *   - weights [G,S,D+S]
 *   - biases [G,S]
 * NB!  gates order is FICO {forget, input, candidate, output}
 *
 * activations is {_f, _g, _h}
 * default: {_f=sigm, _g=tanh, _h=tanh}
 *
 * Equations:
 *
 *   *  - matrix mult
 *  (.) - eltwise mult
 *  [,] - concatenation
 *
 * - ft = _f(Wf*[Ht-1, Xt] + Bf)
 * - it = _f(Wi*[Ht-1, Xt] + Bi)
 * - ct = _g(Wc*[Ht-1, Xt] + Bc)
 * - ot = _f(Wo*[Ht-1, Xt] + Bo)
 * - Ct = ft (.) Ct-1 + it (.) ct
 * - Ht = ot (.) _h(Ct)
 */
class LSTMCell : public RNNCellBase {
public:
    IE_SUPPRESS_DEPRECATED_START
    using RNNCellBase::RNNCellBase;
    using RNNCellBase::operator=;
    IE_SUPPRESS_DEPRECATED_END

    ~LSTMCell() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief GRU Cell layer
 *
 * G - number of gates (=3)
 * N - batch size
 * S - state size (=hidden_size)
 *
 * Inputs:
 *   [N,D] Xt - input data
 *   [N,S] Ht-1 - initial hidden state
 *
 * Outputs:
 *   [N,S] Ht - out hidden state
 *
 * Weights:
 *   - weights [G,S,D+S]
 *   - biases [G,S]
 * NB!  gates order is ZRH {update, reset, output}
 *
 * activations is {_f, _g}
 * default: {_f=sigm, _g=tanh}
 *
 * Equations:
 *
 *   *  - matrix mult
 *  (.) - eltwise mult
 *  [,] - concatenation
 *
 * - zt = _f(Wz*[Ht-1, Xt] + Bz)
 * - rt = _f(Wr*[Ht-1, Xt] + Br)
 * - ht = _g(Wh*[rt (.) Ht-1, Xt] + Bh)
 * - Ht = (1 - zt) (.) ht + zt (.) Ht-1
 */
class GRUCell : public RNNCellBase {
public:
    IE_SUPPRESS_DEPRECATED_START
    using RNNCellBase::RNNCellBase;
    using RNNCellBase::operator=;
    IE_SUPPRESS_DEPRECATED_END

    ~GRUCell() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief RNN Cell layer
 *
 * G - number of gates (=1)
 * N - batch size
 * S - state size (=hidden_size)
 *
 * Inputs:
 *   [N,D] Xt - input data
 *   [N,S] Ht-1 - initial hidden state
 *
 * Outputs:
 *   [N,S] Ht - out hidden state
 *
 * Weights:
 *   - weights [G,S,D+S]
 *   - biases [G,S]
 *
 * activations is {_f}
 * default: {_f=tanh}
 *
 * Equations:
 *
 *   *  - matrix mult
 *  [,] - concatenation
 *
 * - Ht = _f(Wi*[Ht-1, Xt] + Bi)
 */
class RNNCell : public RNNCellBase {
public:
    IE_SUPPRESS_DEPRECATED_START
    using RNNCellBase::RNNCellBase;
    using RNNCellBase::operator=;
    IE_SUPPRESS_DEPRECATED_END

    ~RNNCell() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief Sequence of recurrent cells
 *
 * N  - batch size
 * T  - sequence size
 * S  - state size (=hidden_size)
 * NS - num of state tensors (LSTM=2, GRU/RNN=1)
 * ND - num of direction (BDR=2, WFD/BWD=1)
 *
 * Inputs:
 *   [N,T,D]  Xt - input data
 *   [ND,N,S] Ht-1 - initial hidden state
 *   [ND,N,S] Ct-1 - initial cell state  // if NS==2
 *   [N]      SL - sequence length (optional input)
 *
 * Outputs:
 *   [ND,N,T,S] Xt - input data
 *   [ND,N,S] Ht-1 - initial hidden state
 *   [ND,N,S] Ct-1 - initial cell state  // if NS==2
 *
 * NB! if axis==0 batch and sequense dimensions are swapped (N <-> T) for input and output tensors
 *
 * Weights:
 *   - weights [ND,G,S,D+S]
 *   - biases [ND,G,S]
 * NB! if ND==2 weights are concatenated cell weights [forward_cell_weights, backward_cell_weights]
 *
 */
class RNNSequenceLayer : public RNNCellBase {
public:
    IE_SUPPRESS_DEPRECATED_START
    using RNNCellBase::RNNCellBase;
    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief An axis by which iteration is performed
     *
     * axis=0 means first input/output data blob dimension is sequence
     * axis=1 means first input/output data blob dimension is batch
     */
    unsigned int axis = 1;

    /**
     * @brief Direction of iteration through sequence dimension
     */
    enum Direction {
        FWD, /**< Forward mode. Iterate starts from index 0 with step 1.         */
        BWD, /**< Backward mode. Iterate starts from last index with step -1.    */
        BDR  /**< Bidirectional mode. First is forward pass, second is backward. */
    };

    /** @copybrief Direction */
    Direction direction = FWD;

    ~RNNSequenceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Layer which performs Scale and Shift
 */
class PReLULayer : public WeightableLayer {
public:
    /**
     * @brief A flag that indicates if the same negative_slope value is used for all the features. If false, the value
     * is used pixel wise
     */
    bool _channel_shared = false;

    /**
     * @brief A default constructor. Creates a new PReLULayer instance and initializes layer parameters with the given
     * values.
     *
     * @param prms Initial layer parameters
     */
    IE_SUPPRESS_DEPRECATED_START
    using WeightableLayer::WeightableLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~PReLULayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Power Layer
 *
 * Formula is: output = (offset + scale * input) ^ power
 */
class PowerLayer : public CNNLayer {
public:
    /**
     * @brief An exponent value
     */
    float power = 1.f;
    /**
     * @brief A scale factor
     */
    float scale = 1.f;
    /**
     * @brief An offset value
     */
    float offset = 0.f;

    /**
     * @brief Creates a new PowerLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~PowerLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a Batch Normalization Layer
 */
class BatchNormalizationLayer : public WeightableLayer {
public:
    /**
     * @brief A small value to add to the variance estimate to avoid division by zero
     */
    float epsilon = 1e-3f;

    /**
     * @brief Creates a new BatchNormalizationLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using WeightableLayer::WeightableLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~BatchNormalizationLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a general matrix multiplication operation layer
 *
 * Formula is: dst := alpha*src1*src2 + beta*src3
 */
class GemmLayer : public CNNLayer {
public:
    /**
     * @brief A scale factor of src1 matrix
     */
    float alpha = 1.f;
    /**
     * @brief A scale factor of src3 matrix
     */
    float beta = 1.f;
    /**
     * @brief A flag that indicates if the src1 matrix is to be transposed
     */
    bool transpose_a = false;
    /**
     * @brief A flag that indicates if the src2 matrix is to be transposed
     */
    bool transpose_b = false;
    /**
     * @brief Creates a new GemmLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~GemmLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Pad layer
 *
 * Adds paddings to input tensor
 */
class PadLayer : public CNNLayer {
public:
    /**
     * @enum ePadMode
     * @brief Defines possible modes of pad operation
     */
    enum ePadMode { Constant = 0, Edge, Reflect, Symmetric };

    /**
     * @brief Size of padding in the beginning of each axis
     */
    PropertyVector<unsigned int> pads_begin;
    /**
     * @brief Size of padding in the end of each axis
     */
    PropertyVector<unsigned int> pads_end;
    /**
     * @brief Mode of pad operation
     */
    ePadMode pad_mode = Constant;
    /**
     * @brief A pad value which is used for filling in Constant mode
     */
    float pad_value = 0.0f;
    /**
     * @brief Creates a new PadLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~PadLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Gather layer
 *
 * Gather slices from Dictionary according to Indexes
 */
class GatherLayer : public CNNLayer {
public:
    /**
     * @brief The axis in Dictionary to gather Indexes from
     */
    int axis = 0;
    /**
     * @brief Creates a new GatherLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~GatherLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Strided Slice layer
 *
 * Strided Slice picks from input tensor according parameters
 */
class StridedSliceLayer : public CNNLayer {
public:
    /**
     * @brief The begin_mask is a bitmask where bit i being 0 means
     * to ignore the begin value and instead use the default value
     */
    std::string begin_mask;
    /**
     * @brief Analogous to begin_mask
     */
    std::string end_mask;
    /**
     * @brief The ellipsis_mask is a bitmask where bit i being 1 means
     * the i-th is actually an ellipsis
     */
    std::string ellipsis_mask;
    /**
     * @brief The new_axis_mask_ is a bitmask where bit i being 1 means
     * the i-th position creates a new 1 dimension shape
     */
    std::string new_axis_mask;
    /**
     * @brief The shrink_axis_mask is a bitmask where bit i being 1 means
     * the i-th position shrinks the dimensionality
     */
    std::string shrink_axis_mask;

    /**
     * @brief Creates a new StridedSliceLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~StridedSliceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Shuffle Channels layer
 * Shuffle Channels picks from input tensor according parameters
 */
class ShuffleChannelsLayer : public CNNLayer {
public:
    /**
     * @brief The axis in tensor to shuffle channels
     */
    int axis = 1;

    /**
     * @brief The group of output shuffled channels
     */
    unsigned int group = 1;

    /**
     * @brief Creates a new ShuffleChannelsLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ShuffleChannelsLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Depth To Space layer
 * Depth To Space picks from input tensor according parameters
 */
class DepthToSpaceLayer : public CNNLayer {
public:
    /**
     * @brief The group of output shuffled channels
     */
    unsigned int block_size = 1;

    /**
     * @brief Creates a new DepthToSpaceLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~DepthToSpaceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Space To Depth layer
 * Space To Depth picks from input tensor according parameters
 */
class SpaceToDepthLayer : public CNNLayer {
public:
    /**
     * @brief The group of output Space To Depth
     */
    unsigned int block_size = 1;

    /**
     * @brief Creates a new SpaceToDepthLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SpaceToDepthLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Space To Batch layer
 *
 * Space To Batch picks from input tensor according parameters
 */
class SpaceToBatchLayer : public CNNLayer {
public:
    /**
     * @brief Spatial dimensions blocks sizes
     */
    std::vector<size_t> _block_shape;

    /**
     * @brief Size of padding in the beginning of each axis
     */
    std::vector<size_t> _pads_begin;
    /**
     * @brief Size of padding in the end of each axis
     */
    std::vector<size_t> _pads_end;

    /**
     * @brief Creates a new SpaceToBatchLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SpaceToBatchLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Batch To Space layer
 *
 * Batch To Space picks from input tensor according parameters
 */
class BatchToSpaceLayer : public CNNLayer {
public:
    /**
     * @brief Spatial dimensions blocks sizes
     */
    std::vector<size_t> _block_shape;

    /**
     * @brief It specifies how many elements to crop from the intermediate result
     * across the spatial dimensions
     */
    std::vector<size_t> _crops_begin;

    /**
     * @brief It specifies how many elements to crop from the intermediate result
     * across the spatial dimensions
     */
    std::vector<size_t> _crops_end;

    /**
     * @brief Creates a new BatchToSpaceLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~BatchToSpaceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents SparseFillEmptyRows layer
 *
 * SparseFillEmptyRows fills empty rows in a sparse tensor
 */
class SparseFillEmptyRowsLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new SparseFillEmptyRowsLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SparseFillEmptyRowsLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents SparseSegmentMean(SqrtN, Sum) layers
 * SparseSegmentMean(SqrtN, Sum) layer reduces data along sparse segments of a tensor.
 */
class SparseSegmentReduceLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new SparseSegmentReduceLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SparseSegmentReduceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents ExperimentalSparseWeightedReduce layer
 * ExperimentalSparseWeightedReduce layer reduces data along sparse segments of a tensor.
 */
class ExperimentalSparseWeightedReduceLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new ExperimentalSparseWeightedReduceLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ExperimentalSparseWeightedReduceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents SparseToDense layer
 * SparseToDense layer converts a sparse tensor to a dense tensor.
 */
class SparseToDenseLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new SparseToDenseLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SparseToDenseLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents Bucketize layer
 * Bucketize layer bucketizes the input based on the boundaries.
 */
class BucketizeLayer : public CNNLayer {
public:
    /**
     * @brief Indicates whether the intervals include the right or the left bucket edge.
     */
    bool with_right_bound = true;

    /**
     * @brief Creates a new BucketizeLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~BucketizeLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Reverse Sequence layer
 *
 * Reverse Sequence modifies input tensor according parameters
 */
class ReverseSequenceLayer : public CNNLayer {
public:
    /**
     * @brief The seq_axis dimension in tensor which is partially reversed
     */
    int seq_axis = 1;

    /**
     * @brief The batch_axis dimension in tensor along which reversal is performed
     */
    int batch_axis = 0;

    /**
     * @brief Creates a new ReverseSequence instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ReverseSequenceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a OneHot layer
 * Converts input into OneHot representation.
 */
class OneHotLayer : public CNNLayer {
public:
    /**
     * @brief A depth of representation
     */
    unsigned int depth = 0;

    /**
     * @brief The locations represented by indices in input take value on_value
     */
    float on_value = 1.f;

    /**
     * @brief The locations not represented by indices in input take value off_value
     */
    float off_value = 0.f;

    /**
     * @brief Define the shape of output tensor
     */
    int axis = -1;

    /**
     * @brief Creates a new OneHot instance
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~OneHotLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard RangeLayer layer
 *
 * RangeLayer modifies input tensor dimensions according parameters
 */
class RangeLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new RangeLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~RangeLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Fill layer
 *
 * RFill modifies input tensor according parameters
 */
class FillLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new Fill instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~FillLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a SelectLayer layer
 *
 * SelectLayer layer takes elements from the second (“then”) or the third (“else”) input based on condition mask
 * (“cond”) provided in the first input. The “cond” tensor is broadcasted to “then” and “else” tensors. The output
 * tensor shape is equal to broadcasted shape of “cond”, “then” and “else”.
 */
class SelectLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new SelectLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~SelectLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Broadcast layer
 *
 * Broadcast modifies input tensor dimensions according parameters
 */
class BroadcastLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new Broadcast instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~BroadcastLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a quantization operation layer
 *
 * Element-wise linear quantization of floating point input values into a descrete set of floating point values
 */
class QuantizeLayer : public CNNLayer {
public:
    /**
     * @brief The number of quantization levels
     */
    size_t levels = 1;

    /**
     * @brief Creates a new QuantizeLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~QuantizeLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Math layers
 *
 * Math modifies input tensor dimensions according parameters
 */
class MathLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new Math instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~MathLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard Reduce layers
 *
 * Reduce modifies input tensor according parameters
 */
class ReduceLayer : public CNNLayer {
public:
    /**
     * @brief The keep_dims dimension in tensor which is partially reversed
     */
    bool keep_dims = true;

    /**
     * @brief Creates a new Reduce instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ReduceLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard TopK layer
 *
 * TopK picks top K values from input tensor according parameters
 */
class TopKLayer : public CNNLayer {
public:
    /**
     * @brief The mode could be 'max' or 'min'
     */
    std::string mode;
    /**
     * @brief top K values sort mode could be 'value' or 'index'
     */
    std::string sort;
    /**
     * @brief The axis dimension in tensor which is top K values are picked
     */
    int axis = -1;

    /**
     * @brief Creates a new TopKLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~TopKLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents Unique layer.
 *
 * The Unique operation searches for unique elements in 1-D input
 */
class UniqueLayer : public CNNLayer {
public:
    /**
     * @brief A flag indicating whether to sort unique elements
     */
    bool sorted;
    /**
     * @brief A flag indicating whether to return indices of input data elements in the output of uniques
     */
    bool return_inverse;
    /**
     * @brief A flag indicating whether to return a number of occurrences for each unique element
     */
    bool return_counts;

    /**
     * @brief Creates a new UniqueLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~UniqueLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard NonMaxSuppression layer
 */
class NonMaxSuppressionLayer : public CNNLayer {
public:
    /**
     * @brief The 'center_point_box' indicates the format of the box data
     */
    bool center_point_box = false;
    /**
     * @brief The 'sort_result_descending' indicates that result will sort descending by score through all batches and
     * classes
     */
    bool sort_result_descending = true;
    /**
     * @brief Output type for first and third inputs
     */
    std::string output_type = "I64";
    /**
     * @brief Creates a new NonMaxSuppressionLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~NonMaxSuppressionLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard ScatterUpdate layer
 */
class ScatterUpdateLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new ScatterUpdateLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ScatterUpdateLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents a standard ScatterElementsUpdate layer
 */
class ScatterElementsUpdateLayer : public CNNLayer {
public:
    /**
     * @brief Creates a new ScatterElementsUpdateLayer instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    ~ScatterElementsUpdateLayer() override;
};

/**
 * @deprecated Migrate to IR v10 and work with ngraph::Function directly. The method will be removed in 2021.1
 * @brief This class represents an onnx ExperimentalDetectronPriorGridGenerator Layer
 */
class ExperimentalDetectronPriorGridGeneratorLayer : public CNNLayer {
public:
    /**
     * @brief flatten value
     */
    int flatten = 1;
    /**
     * @brief Value of grid width
     */
    int grid_w = 0;
    /**
     * @brief Value of grid height
     */
    int grid_h = 0;
    /**
     * @brief Value of width step between grid cells
     */
    float stride_w = 0.f;
    /**
     * @brief Value of height step between grid cells
     */
    float stride_h = 0.f;

    /**
     * @brief Creates a new ExperimentalDetectronPriorGridGenerator instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    virtual ~ExperimentalDetectronPriorGridGeneratorLayer();
};

/**
 * @brief This class represents a standard ExperimentalDetectronTopKROIs layer
 */
class ExperimentalDetectronTopKROIs : public CNNLayer {
public:
    /**
     * @brief The maximum number of output rois
     */
    int max_rois = 0;
    /**
     * @brief Creates a new ExperimentalDetectronTopKROIs instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    virtual ~ExperimentalDetectronTopKROIs();
};

/**
 * @brief This class represents an onnx ExperimentalDetectronGenerateProposalsSingleImage Layer
 */
class ExperimentalDetectronGenerateProposalsSingleImageLayer : public CNNLayer {
public:
    /**
     * @brief Minimium width and height for boxes
     */
    float min_size = 0.f;
    /**
     * @brief Non max suppression threshold
     */
    float nms_threshold = 0.7f;
    /**
     * @brief Maximum number of anchors selected before nms
     */
    int pre_nms_topn = 1000;
    /**
     * @brief Maximum number of anchors selected after nms
     */
    int post_nms_topn = 1000;

    /**
     * @brief Creates a new ExperimentalDetectronGenerateProposalsSingleImage instance.
     */
    IE_SUPPRESS_DEPRECATED_START
    using CNNLayer::CNNLayer;
    IE_SUPPRESS_DEPRECATED_END

    virtual ~ExperimentalDetectronGenerateProposalsSingleImageLayer();
};

_IE_SUPPRESS_DEPRECATED_END_MSVC

}  // namespace InferenceEngine
