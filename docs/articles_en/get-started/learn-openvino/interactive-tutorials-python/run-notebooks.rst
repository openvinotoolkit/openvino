Run Notebooks
===============================================================================================

.. meta::
   :description: Run Jupyter notebooks to go through the tutorials and learn OpenVINO™ toolkit.


This guide will show you how to launch and manage OpenVINO™ Notebooks on your machine.
Before you proceed, make sure you have followed the
:doc:`installation steps <notebooks-installation>`.


Launch a Single Notebook
###############################################################################################

If you want to launch only one notebook, such as the *Monodepth* notebook, run the
command below.

.. code:: bash

   jupyter lab notebooks/vision-monodepth/vision-monodepth.ipynb

Launch All Notebooks
###############################################################################################

.. code:: bash

   jupyter lab notebooks

In your browser, select a notebook from the file browser in Jupyter Lab, using the left
sidebar. Each tutorial is located in a subdirectory within the ``notebooks`` directory.

|launch-jupyter|


Manage the Notebooks
###############################################################################################

Shut Down Jupyter Kernel
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To end your Jupyter session, press ``Ctrl-c``. This will prompt you to
``Shutdown this Jupyter server (y/[n])?`` enter ``y`` and hit ``Enter``.

Deactivate Virtual Environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

First, make sure you use the terminal window where you activated ``openvino_env``.
To deactivate your ``virtualenv``, simply run:

.. code:: bash

   deactivate

This will deactivate your virtual environment.

Reactivate Virtual Environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To reactivate your environment, run:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code:: bash

         source openvino_env\Scripts\activate

   .. tab-item:: Linux
      :sync: linux

      .. code:: bash

         source openvino_env/bin/activate

   .. tab-item:: macOS
      :sync: macos

      .. code:: bash

         source openvino_env/bin/activate


Then type ``jupyter lab`` or ``jupyter notebook`` to launch the notebooks again.

Delete Virtual Environment
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This operation is optional. However, if you want to remove your virtual environment,
simply delete the ``openvino_env`` directory:

.. tab-set::

   .. tab-item:: Windows
      :sync: windows

      .. code:: bash

         rmdir /s openvino_env

   .. tab-item:: Linux
      :sync: linux

      .. code:: bash

         rm -rf openvino_env

   .. tab-item:: macOS
      :sync: macos

      .. code:: bash

         rm -rf openvino_env


Remove openvino_env Kernel from Jupyter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: bash

   jupyter kernelspec remove openvino_env


If you run into issues, check the `Troubleshooting <#-troubleshooting>`__, and
`FAQs <#-faq>`__ sections or start a GitHub
`discussion <https://github.com/openvinotoolkit/openvino_notebooks/discussions>`__.

.. |launch-jupyter| image:: https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif

Additional Resources
####################

* `OpenVINO™ Notebooks - Github Repository <https://github.com/openvinotoolkit/openvino_notebooks/>`_
