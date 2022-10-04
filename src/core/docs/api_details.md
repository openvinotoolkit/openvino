# OpenVINO Core API

OpenVINO Core API contains two folders:
 * [ngraph](../include/ngraph/) - is a legacy API, this API is not developed more. Only aliases to new operation and operation sets extend this API.
 * [openvino](../include/openvino/) - current public API, this part will be described below.

## Structure of Core API
<pre>
 <code>
 <a href="../include/openvino">openvino/</a>                  // Common folder with OpenVINO 2.0 API
    <a href="../include/openvino/core/">core/</a>                   // Contains common classes which are responsible for model representation
    op/                     // Contains all supported OpenVINO operations
    opsets/                 // Contains definitions of each official OpenVINO opset
    pass/                   // Defines classes for developing transformation and several common transformations
    runtime/                // Contains OpenVINO tensor definition
 </code>
</pre>



## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
