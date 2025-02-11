Plugin Properties
=================


.. meta::
   :description: Use the ov::Property class to define access rights and
                 specific properties of an OpenVINO plugin.


Plugin can provide own device-specific properties.

Property Class
##############

OpenVINO API provides the interface ov::Property which allows to define the property and access rights. Based on that, a declaration of plugin specific properties can look as follows:

.. doxygensnippet:: src/plugins/template/include/template/properties.hpp
   :language: cpp
   :fragment: [properties:public_header]


