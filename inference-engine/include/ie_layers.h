// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for internal Layers structure to describe layers information
 * @file ie_layers.h
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iterator>
#include <cctype>
#include "ie_common.h"
#include "ie_data.h"
#include "ie_blob.h"
#include "ie_device.hpp"

namespace InferenceEngine {
/**
 * @brief This is an internal common Layer parameter parsing arguments
 */
struct LayerParams {
    /// @brief Layer name
    std::string name;
    /// @brief Layer type
    std::string type;
    /// @brief Layer precision
    Precision precision;
};

/**
 * @brief This is a base abstraction Layer - all DNN Layers inherit from this class
 */
class CNNLayer  {
public:
    /**
     * @brief A shared pointer to CNNLayer
     */
    using  Ptr = std::shared_ptr<CNNLayer>;

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
    Ptr _fusedWith;
    /**
     * @brief Convenience user values to store in this object as extra data
     */
    UserValue userValue;
    /**
     * @brief Layer affinity set by user.
     */
    std::string affinity;

    /**
     * @brief A constructor. Creates a new CNNLayer instance and initializes layer parameters with the given values.
     * @param prms Basic common parsing parameters
     */
    explicit CNNLayer(const LayerParams &prms) : name(prms.name), type(prms.type),
                                                 precision(prms.precision), userValue({0}) {
    }

    /**
     * @brief A virtual destructor
     */
    virtual ~CNNLayer() = default;

    /**
     * @brief Sets a layer to be fused with
     * @param layer Reference to the layer to be fused with
     */
    void fuse(Ptr &layer) {
        _fusedWith = layer;
    }

    /**
     * @brief Returns the first element of the input data for this layer
     * @return A smart pointer to the input data element
     */
    virtual const DataPtr input() const {
        if (insData.empty()) {
            THROW_IE_EXCEPTION << "Internal error: input data is empty";
        }
        auto lockedFirstInsData = insData[0].lock();
        if (!lockedFirstInsData) {
            THROW_IE_EXCEPTION << "Internal error: unable to lock weak_ptr\n";
        }
        return lockedFirstInsData;
    }

    /**
     * @brief Checks if the input data and layer data are legitimate
     */
    INFERENCE_ENGINE_API_CPP(void) validateLayer();

    /**
     * @brief Gets float value for the given parameter
     * @param param - name of the parameter to find
     * @param def - default value of the parameter if not found
     * @return float value
     */
    float GetParamAsFloat(const char* param, float def) const {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a float value for the given layer parameter
     * @param param Name of the layer parameter
     * @return A float value for the specified parameter
     */
    float GetParamAsFloat(const char *param) const {
        std::string val = GetParamAsString(param);
        try {
            return std::stof(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to float.";
        }
    }

    /**
     * @brief Returns a vector of float values for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char *param, std::vector<float> def) const {
        std::string vals = GetParamAsString(param, "");
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of float values for the given parameter
     * @param param Name of the layer parameter
     * @return vector of float values
     */
    std::vector<float> GetParamAsFloats(const char *param) const {
        std::string vals = GetParamAsString(param);
        std::vector<float> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stof(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to floats.";
            }
        }
        return result;
    }

    /**
     * @brief Returns an integer value for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char *param, int def) const {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to int.";
        }
    }

    /**
     * @brief Returns an integer value for the given parameter
     * @param param Name of the layer parameter
     * @return An int value for the specified parameter
     */
    int GetParamAsInt(const char *param) const {
        std::string val = GetParamAsString(param);
        try {
            return std::stoi(val);
        } catch (...) {
            THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer " << name
                               << ". Value " << val << " cannot be casted to int.";
        }
    }


    /**
     * @brief Returns a vector of int values for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of int values
     */
    std::vector<int> GetParamAsInts(const char *param, std::vector<int> def) const {
        std::string vals = GetParamAsString(param, "");
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                                   << ". Value " << vals << " cannot be casted to int.";
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of int values for the given parameter
     * @param param Name of the layer parameter
     * @return vector of int values
     */
    std::vector<int> GetParamAsInts(const char *param) const {
        std::string vals = GetParamAsString(param);
        std::vector<int> result;
        std::istringstream stream(vals);
        std::string str;
        while (getline(stream, str, ',')) {
            try {
                result.push_back(std::stoi(str));
            } catch (...) {
                THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " " << str << " from IR for layer " << name
                        << ". Value " << vals <<  " cannot be casted to int.";
            }
        }
        return result;
    }
    /**
     * @brief Returns an unsigned integer value for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int GetParamAsUInt(const char *param, unsigned int def) const {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name
                              + ". Value " + val + " cannot be casted to int.";
        try {
            int value = std::stoi(val);
            if (value < 0) {
                THROW_IE_EXCEPTION << message;
            }
            return static_cast<unsigned int>(value);
        } catch (...) {
            THROW_IE_EXCEPTION << message;
        }
    }

    /**
     * @brief Returns an unsigned integer value for the given parameter
     * @param param Name of the layer parameter
     * @return An unsigned integer value for the specified parameter
     */
    unsigned int GetParamAsUInt(const char *param) const {
        std::string val = GetParamAsString(param);
        std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer " + name
                                                 + ". Value " + val + " cannot be casted to int.";
        try {
            int value = std::stoi(val);
            if (value < 0) {
                THROW_IE_EXCEPTION << message;
            }
            return static_cast<unsigned int>(value);
        } catch (...) {
            THROW_IE_EXCEPTION << message;
        }
    }


    /**
     * @brief Returns a vector of unsigned int values for the given parameter or returns the default value
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> GetParamAsUInts(const char *param, std::vector<unsigned int> def) const {
        std::string vals = GetParamAsString(param, "");
        std::vector<unsigned int> result;
        std::istringstream stream(vals);
        std::string str;
        std::string message = "Cannot parse parameter " + std::string(param) + " " + str + " from IR for layer " + name
                              + ". Value " + vals +  " cannot be casted to int.";
        if (vals.empty())
            return def;
        while (getline(stream, str, ',')) {
            try {
                int value = std::stoi(str);
                if (value < 0) {
                    THROW_IE_EXCEPTION << message;
                }
                result.push_back(static_cast<unsigned int>(value));
            } catch (...) {
                THROW_IE_EXCEPTION << message;
            }
        }
        return result;
    }

    /**
     * @brief Returns a vector of unsigned int values for the given parameter
     * @param param Name of the layer parameter
     * @return vector of unsigned int values
     */
    std::vector<unsigned int> GetParamAsUInts(const char *param) const {
        std::string vals = GetParamAsString(param);
        std::vector<unsigned int> result;
        std::istringstream stream(vals);
        std::string str;
        std::string message = "Cannot parse parameter " + std::string(param) + " " + str + " from IR for layer " + name
                                                        + ". Value " + vals +  " cannot be casted to int.";
        while (getline(stream, str, ',')) {
            try {
                int value = std::stoi(str);
                if (value < 0) {
                    THROW_IE_EXCEPTION << message;
                }
                result.push_back(static_cast<unsigned int>(value));
            } catch (...) {
                THROW_IE_EXCEPTION << message;
            }
        }
        return result;
    }
    /**
     * @brief Returns an boolean value for the given parameter.
     * The valid values are (true, false, 1, 0).
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return An bool value for the specified parameter
     */
    bool GetParamsAsBool(const char *param, bool def) const {
        std::string val = GetParamAsString(param, std::to_string(def).c_str());
        std::string loweredCaseValue;
        std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
            return std::tolower(value);
        });

        bool result = false;

        if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
            // attempting parse using non alpha bool
            return static_cast<bool>(GetParamAsInt(param, def));
        }

        return result;
    }

    /**
     * @brief Returns a string value for the given parameter or returns the default one
     * @param param Name of the layer parameter
     * @param def Default value of the parameter if not found
     * @return A string value
     */
    std::string GetParamAsString(const char *param, const char *def) const {
        auto it = params.find(param);
        if (it == params.end()) {
            return def;
        }
        return (*it).second;
    }

    /**
     * @brief Returns a string value for the given parameter.
     * Throws exception if parameter was not found.
     * @param param Name of the layer parameter
     * @return A string value
     */
    std::string GetParamAsString(const char *param) const {
        auto it = params.find(param);
        if (it == params.end()) {
            THROW_IE_EXCEPTION << "No such parameter name '" << param << "' for layer " << name;
        }
        return (*it).second;
    }

    /**
     * @brief Map of pairs: (parameter name, parameter value)
     */
    std::map<std::string, std::string> params;
    /**
     * @brief Map of pairs: (name, weights/biases blob)
     */
    std::map<std::string, Blob::Ptr> blobs;
};

/**
 * @brief Alias for CNNLayer object
 */
using GenericLayer = class CNNLayer;

/**
 * @brief This class represents a layer with Weights and/or Biases (e.g. Convolution/Fully Connected, etc.)
 */
class WeightableLayer : public CNNLayer {
public:
    /**
     * @brief A default constructor. Constructs a WeightableLayer instance and initiates layer parameters with the given values
     * @param prms Initial layer parameters
     */
    explicit WeightableLayer(const LayerParams &prms) : CNNLayer(prms) {}

    /**
     * @brief A pointer to a weights blob
     */
    Blob::Ptr _weights;
    /**
     * @brief A pointer to a biases blob
     */
    Blob::Ptr _biases;

    /**
     * @brief Constructs a WeightableLayer instance and initiates layer parameters with the given values
     */
    using CNNLayer::CNNLayer;
};

/**
 * @brief This class represents a standard 3D Convolution Layer
 */
class ConvolutionLayer : public WeightableLayer {
public:
    /**
     * @brief A convolution kernel width
     */
    unsigned int _kernel_x = 0;
    /**
     * @brief A convolution kernel height
     */
    unsigned int _kernel_y = 0;
    /**
     * @brief An input convolution stride width
     */
    unsigned int _stride_x = 1;
    /**
     * @brief An Input convolution stride height
     */
    unsigned int _stride_y = 1;
    /**
     * @brief A number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth = 0;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x = 0;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y = 0;
    /**
     * @brief Dilation width
     */
    unsigned int _dilation_x = 1;
    /**
     * @brief Dilation height
     */
    unsigned int _dilation_y = 1;
    /**
     * @brief Number of groups
     */
    unsigned int _group = 1;

    /**
     * @brief Creates a new ConvolutionLayer instance.
     */
    using WeightableLayer::WeightableLayer;
};

/**
 * @brief This class represents a standard deconvolution layer
 */
class DeconvolutionLayer : public WeightableLayer {
public:
    /**
     * @brief Deconvolution kernel width
     */
    unsigned int _kernel_x = 0;
    /**
     * @brief Deconvolution kernel height
     */
    unsigned int _kernel_y = 0;
    /**
     * @brief Input Deconvolution stride width
     */
    unsigned int _stride_x = 0;
    /**
     * @brief Input Deconvolution stride height
     */
    unsigned int _stride_y = 0;
    /**
     * @brief number of output feature maps (size) generating the 3'rd output dimension
     */
    unsigned int _out_depth = 0;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x = 0;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y = 0;
    /**
     * @brief Dilation width
     */
    unsigned int _dilation_x = 0;
    /**
     * @brief Dilation height
     */
    unsigned int _dilation_y = 0;
    /**
     * @brief Number of groups
     */
    unsigned int _group = 0;

    /**
     * @brief Creates a new DeconvolutionLayer instance.
     */
    using WeightableLayer::WeightableLayer;
};

/**
 * @brief This class represents a standard pooling layer
 */
class PoolingLayer : public CNNLayer {
public:
    /**
     * @brief Pooling kernel width
     */
    unsigned int _kernel_x = 0;
    /**
     * @brief Pooling kernel height
     */
    unsigned int _kernel_y = 0;
    /**
     * @brief Input Pooling stride width
     */
    unsigned int _stride_x = 0;
    /**
     * @brief Input Pooling stride height
     */
    unsigned int _stride_y = 0;
    /**
     * @brief Input padding width
     */
    unsigned int _padding_x = 0;
    /**
     * @brief Input padding height
     */
    unsigned int _padding_y = 0;

    /**
     * @enum PoolType
     * @brief Defines available pooling types
     */
    enum PoolType {
        MAX = 1,
        AVG = 2,
        STOCH = 3,
        ROI = 4,
        SPACIAL_PYRAMID = 5
    };

    /**
     * @brief A pooling type
     */
    PoolType _type = MAX;

    /**
     * @brief A flag that indicates if padding is excluded or not
     */
    bool _exclude_pad = false;

    /**
    * @brief Creates a new PoolingLayer instance.
    */
    using CNNLayer::CNNLayer;
};

/**
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
    using WeightableLayer::WeightableLayer;
};

/**
 * @brief This class represents concatenation layer
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
    * If batch is used, then batch needs to be specified as an input dimension also
    * In current implementation 1 means channels, 0 - batch
    */
    using CNNLayer::CNNLayer;
};

/**
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
    using CNNLayer::CNNLayer;
};

/**
 * @brief This class represents a Linear Response Normalization (LRN) Layer
 */
class NormLayer : public CNNLayer {
public:
    /**
     * @brief Response size
     */
    unsigned int _size = 0;
    /**
     * @deprecated
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
    using CNNLayer::CNNLayer;
};

/**
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
    using CNNLayer::CNNLayer;
};

/**
 * @class GRNLayer
 * @brief This class represents standard GRN Layer
 */
class GRNLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new GRNLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit GRNLayer(const LayerParams &prms) : CNNLayer(prms), bias(0.f) {}

    /**
     * @brief Bias for squares sum
     */
    float bias = 0.f;
};

/**
 * @class MVNLayer
 * @brief This class represents standard MVN Layer
 */
class MVNLayer : public CNNLayer {
public:
    /**
    * @brief A default constructor. Creates a new MVNLayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit MVNLayer(const LayerParams &prms) : CNNLayer(prms), across_channels(0), normalize(1) {}

    /**
     * @brief Indicate that mean value is calculated across channels
     */
    int across_channels;

    /**
    * @brief Indicate that the result needs to be normalized
    */
    int normalize = 1;
};

/**
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
    using CNNLayer::CNNLayer;
};

/**
 * @brief This class represents a Clamp activation layer
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
    using CNNLayer::CNNLayer;
};

/**
 * @brief This class represents an element wise operation layer
 */
class EltwiseLayer : public CNNLayer {
public:
    /**
     * @enum eOperation
     * @brief Defines possible operations that can be used
     */
    enum eOperation {
        Sum = 0, Prod, Max
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
    using CNNLayer::CNNLayer;
};

/**
 * @brief This class represents a standard crop layer
 */
class CropLayer : public CNNLayer {
public:
    /**
     * @brief A vector of dimensions for cropping
     */
    std::vector<int> axis;
    /**
     * @deprecated result size is defined by second input
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
    using CNNLayer::CNNLayer;
};

/**
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
    using CNNLayer::CNNLayer;
};

/**
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
    using CNNLayer::CNNLayer;
};


/**
 * @brief This class represents a Layer which performs Scale and Shift
 */
class ScaleShiftLayer : public WeightableLayer {
public:
    /**
     * @brief A flag that indicates if the same value is used for all the features. If false, the value is used pixel wise
     */
    unsigned int _broadcast = 0;

    /**
     * @brief Creates a new ScaleShiftLayer instance.
     */
    using WeightableLayer::WeightableLayer;
};

/**
* @class PReLULayer
* @brief This class represents a Layer which performs Scale and Shift
*/
class PReLULayer : public WeightableLayer {
public:
    /**
     * @brief A flag that indicates if the same negative_slope value is used for all the features. If false, the value is used pixel wise
     */
    bool _channel_shared;

public:
    /**
    * @brief A default constructor. Creates a new PReLULayer instance and initializes layer parameters with the given values.
    * @param prms Initial layer parameters
    */
    explicit PReLULayer(const LayerParams &prms) : WeightableLayer(prms), _channel_shared(false) {}
};

/**
 * @brief This class represents a standard Power Layer
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
    using CNNLayer::CNNLayer;
};

/**
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
    using WeightableLayer::WeightableLayer;
};

}  // namespace InferenceEngine
