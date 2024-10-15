Install Intel® Distribution of OpenVINO™ Toolkit from PyPI Repository
========================================================================


.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Windows, Linux, and
                 macOS operating systems, using a PyPi package.


.. note::

   Note that the PyPi distribution:

   * offers the Python API only
   * is dedicated to users of all major OSes: Windows, Linux, and macOS
     (all x86_64 / arm64 architectures)
   * macOS offers support only for CPU inference


.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | :doc:`System Requirements Page <../../../about-openvino/release-notes-openvino/system-requirements>`
      | `PyPI OpenVINO page <https://pypi.org/project/openvino/>`__


   .. tab-item:: Processor Notes
      :sync: processor-notes

      | To see if your processor includes the integrated graphics technology and supports iGPU
        inference, refer to:
      | `Product Specifications <https://ark.intel.com/>`__




Installing OpenVINO Runtime
###########################

Step 1. Set Up Python Virtual Environment
+++++++++++++++++++++++++++++++++++++++++

Use a virtual environment to avoid dependency conflicts.
To create a virtual environment, use the following command:

.. tab-set::

    .. tab-item:: Windows
       :sync: windows

       .. code-block:: sh

          python -m venv openvino_env

    .. tab-item:: Linux and macOS
       :sync: linux-and-macos

       .. code-block:: sh

          python3 -m venv openvino_env


Step 2. Activate Virtual Environment
++++++++++++++++++++++++++++++++++++


.. tab-set::

    .. tab-item:: Windows
       :sync: windows

       .. code-block:: sh

          openvino_env\Scripts\activate

    .. tab-item:: Linux and macOS
       :sync: linux-and-macos

       .. code-block:: sh

          source openvino_env/bin/activate


.. important::

   The above command must be re-run every time a new command terminal window is opened.


Step 3. Set Up and Update PIP to the Highest Version
++++++++++++++++++++++++++++++++++++++++++++++++++++

Use the following command:

.. code-block:: sh

   python -m pip install --upgrade pip


Step 4. Install the Package
+++++++++++++++++++++++++++

Use the following command to install OpenVINO Base or OpenVINO GenAI package:

.. code-block:: python

   python -m pip install openvino

Step 5. Verify that the Package Is Installed
++++++++++++++++++++++++++++++++++++++++++++

Run the command below:

.. code-block:: sh

   python -c "from openvino import Core; print(Core().available_devices)"

If installation was successful, you will see the list of available devices.


Congratulations! You've just Installed OpenVINO! For some use cases you may still
need to install additional components. Check the
:doc:`list of additional configurations <../configurations>`
to see if your case needs any of them.






| **Simplified Build and Integration**
|   The package includes CMake configurations, precompiled static libraries, and headers, which
    can be easily accessed through the Python API. You can use the `get_cmake_path()` method to
    retrieve the paths to the CMake configurations and libraries:

.. code-block:: python

   from openvino import get_cmake_path
   cmake_path = get_cmake_path()

For detailed instructions on how to use these configurations in your build setup, check out the
:ref:`Create a library with extensions <create_a_library_with_extensions>` section.







What's Next?
####################

Now that you've installed OpenVINO Runtime, you're ready to run your own machine learning
applications! Learn more about how to integrate a model in OpenVINO applications by trying out
the following tutorials.

.. image:: https://user-images.githubusercontent.com/15709723/127752390-f6aa371f-31b5-4846-84b9-18dd4f662406.gif
   :width: 400

Try the `Python Quick Start Example <https://docs.openvino.ai/2024/notebooks/vision-monodepth-with-output.html>`__
to estimate depth in a scene using an OpenVINO monodepth model in a Jupyter Notebook inside
your web browser.


Get started with Python
+++++++++++++++++++++++

Visit the :doc:`Tutorials <../../../learn-openvino/interactive-tutorials-python>` page for more
Jupyter Notebooks to get you started with OpenVINO, such as:

* `OpenVINO Python API Tutorial <https://docs.openvino.ai/2024/notebooks/openvino-api-with-output.html>`__
* `Basic image classification program with Hello Image Classification <https://docs.openvino.ai/2024/notebooks/hello-world-with-output.html>`__
* `Convert a PyTorch model and use it for image background removal <https://docs.openvino.ai/2024/notebooks/vision-background-removal-with-output.html>`__



Additional Resources
####################

- Intel® Distribution of OpenVINO™ `toolkit home page <https://software.intel.com/en-us/openvino-toolkit>`__
