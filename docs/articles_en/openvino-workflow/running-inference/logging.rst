Logging Configuration
=====================

.. meta::
   :description: Learn how to configure OpenVINO™ log message handling
                 using the set_log_callback and reset_log_callback APIs.


OpenVINO™ Runtime provides a public C++ API for customizing how log messages
are handled. By default, log messages are printed to ``std::cout``. You can
redirect, filter, buffer, or suppress log output by setting a custom callback.

.. note::

   The log callback API is currently available in the C++ API only.
   Python bindings are not yet provided.


Setting a Custom Log Callback
#############################

Use ``ov::util::set_log_callback()`` to register a function that will receive
all OpenVINO log messages. The callback must accept a single
``std::string_view`` argument:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_logging.cpp
         :language: cpp
         :fragment: [ov:logging:part0]


Disabling Log Output
#####################

To suppress all log output, pass an empty (default-constructed) callable.
This can be useful in production deployments where log output is not needed:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_logging.cpp
         :language: cpp
         :fragment: [ov:logging:part1]


Thread-Safe Log Collection
##########################

When using OpenVINO in a multi-threaded application, make sure the callback
is thread-safe. The following example shows how to collect log messages into
a buffer using a mutex:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_logging.cpp
         :language: cpp
         :fragment: [ov:logging:part2]


API Reference
#############

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``ov::util::set_log_callback(callback)``
     - Sets a user-defined log message handler. Pass an empty callable
       to disable logging entirely.
   * - ``ov::util::reset_log_callback()``
     - Resets the log handler to the default (``std::cout``).

Both functions are declared in ``<openvino/core/log.hpp>``.

See Also
########

- :doc:`Hello Classification Sample <../../get-started/learn-openvino/openvino-samples/hello-classification>` — uses ``set_log_callback`` as a real-world example.
- :doc:`Integrate OpenVINO™ with Your Application <running-inference>`
