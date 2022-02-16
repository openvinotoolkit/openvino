# Extension Library {#openvino_docs_Extensibility_UG_Extension}

OpenVINO™ provides an `OPENVINO_CREATE_EXTENSIONS()` macro, which allows to define an entry point to a library with OpenVINO™ Extensions.
This macro should have a vector of all OpenVINO™ Extensions as an argument.

Based on that, the declaration of an extension class can look as follows:

@snippet template_extension/new/ov_extension.cpp ov_extension:entry_point
