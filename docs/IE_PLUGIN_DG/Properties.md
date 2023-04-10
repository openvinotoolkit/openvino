# Plugin Properties {#openvino_docs_ov_plugin_dg_properties}

Plugin can provide own device specific properties.

Property Class
------------------------

OpenVINO API provides the interface ov::Property which allows to define the property and access rights. Based on that, a declaration of plugin specific properties can look as follows: 

@snippet include/template/properties.hpp properties:public_header
