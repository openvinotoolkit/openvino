OpenVINO Integration with Ubuntu Snap
=============================================

A snap is a way to containerize applications and embed them on Linux devices. Currently, OpenVINO
supports this form of deployment for Ubuntu. Building a snap package involves several steps,
including setting up your development environment, creating the necessary configuration files, and
`using the snapcraft tool to build the Snap <https://snapcraft.io/docs/creating-a-snap>`__.
This article will show you how to integrate OpenVINO toolkit with your application snap:

* `Method 1: User Application Snap based on OpenVINO Sources <#method-1-user-application-snap-based-on-openvino-sources>`__
* `Method 2: Separate OpenVINO and User Application Snaps <#method-2-separate-openvino-and-user-application-snaps>`__
* `Method 3: User Application Snap based on OpenVINO Debian Packages (Recommended) <#method-3-recommended-user-application-snap-based-on-openvino-debian-packages>`__



Method 1: User Application Snap based on OpenVINO Sources
#########################################################

OpenVINO libraries can be built using the CMake plugin (https://snapcraft.io/docs/cmake-plugin).
To build and install it to the application snap image, you need to configure the new part in
the application snapcraft.yaml:

.. code-block:: sh

   openvino-build:
     plugin: cmake
     source-type: git
     source: https://github.com/openvino.git
     source-branch: master
     cmake-generator: Ninja
     cmake-parameters:
       - -DENABLE_SAMPLES:BOOL=OFF
       - -DENABLE_TESTS:BOOL=OFF
     build-environment:
       - CMAKE_BUILD_PARALLEL_LEVEL: ${SNAPCRAFT_PARALLEL_BUILD_COUNT}
       - CMAKE_BUILD_TYPE: Release
     build-packages:
       - build-essential
       - ninja-build
       - pkg-config
       - gzip



Method 2: Separate OpenVINO and User Application Snaps
######################################################

This approach means that OpenVINO libraries and user applications will be distributed as
separate snaps. It involves three steps:

1. Configure the OpenVINO snapcraft.yaml.

   Add the part to build and install OpenVINO as described in the 1st Method:

     .. code-block:: sh

        openvino-build:
          plugin: cmake
          source-type: git
          source: https://github.com/openvino.git
          source-branch: master
          cmake-generator: Ninja
          cmake-parameters:
            - -DENABLE_SAMPLES:BOOL=OFF
            - -DENABLE_TESTS:BOOL=OFF
          build-environment:
            - CMAKE_BUILD_PARALLEL_LEVEL: ${SNAPCRAFT_PARALLEL_BUILD_COUNT}
            - CMAKE_BUILD_TYPE: Release
          build-packages:
            - build-essential
            - ninja-build
            - pkg-config
            - gzip

   Define the slots provided by the OpenVINO Snap. Slots are the interfaces your Snap
   exposes for other Snaps to connect to:

     .. code-block:: sh

        slots:
          openvino-libs:
            interface: content
            content: openvino-libs
            read:
              - $SNAP/usr/local/li
          openvino-3rdparty-libs:
            interface: content
            content: openvino-extra-libs
            read:
              - $SNAP/usr/local/runtime/3rdparty/tbb/lib

2. Configure the application's snapcraft.yaml:

   Edit your snapcraft.yaml file to include OpenVINO plugs.

   .. code-block:: sh

      plugs:
        openvino-libs:
          interface: content
          content: openvino-libs
          target: $SNAP/openvino-libs
          default-provider: openvino-libs-test

        openvino-3rdparty-libs:
          interface: content
          content: openvino-extra-libs
          target: $SNAP/openvino-extra-libs
          default-provider: openvino-libs-test

   Add OpenVINO snap to build-snaps:

   .. code-block:: sh

      parts:
        app-build:
          build-snaps:
            - openvino-libs-test

   Set the OpenVINO environment in the build part:

   .. code-block:: sh

      parts:
       app-build:
         build-environment:
           - OpenVINO_DIR: /snap/openvino-libs/current/usr/local/runtime/cmake
           - LD_LIBRARY_PATH: $LD_LIBRARY_PATH:/snap/openvino-libs/current/usr/local/runtime/3rdparty/tbb/lib


   Set the OpenVINO environment in the apps section:

   .. code-block:: sh

      apps:
        app:
          command: usr/local/app
          environment:
            LD_LIBRARY_PATH: $LD_LIBRARY_PATH:$SNAP/openvino-libs:$SNAP/openvino-extra-libs

3. Install snaps and Connect plugs. Snaps can be connected automatically only if they are
   published by the same user, otherwise you need to manually connect Application plugs with
   OpenVINO slots after installation:

   .. code-block:: sh

      snap connect app:openvino-libs openvino-libs:openvino-libs
      snap connect app:openvino-3rdparty-libs openvino-libs:openvino-3rdparty-libs


Method 3 (Recommended): User Application Snap based on OpenVINO Debian Packages
###############################################################################

OpenVINO toolkit is also distributed via the
`APT repository <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html>`__,
which can be used in the snaps. Third-party apt repositories can be added to the snap's
snapcraft.yaml (`see the snapcraft guide <https://snapcraft.io/docs/package-repositories>`__).

1. Download the `GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB <https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html#:~:text=Install%20the%20GPG,SW%2DPRODUCTS.PUB>`__:

   .. code-block:: sh

      wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

2. To determine a key-id from a given key file with gpg, type the following:

   .. code-block:: sh

      gpg --show-keys ./GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

      pub   rsa2048 2019-07-26 [SC] [expired: 2023-07-26]
            E1BA4ECEFB0656C61BF9794936B9569B3F1A1BC7
      uid                      KEY-PIDT-PGP-20190726

      pub   rsa2048 2020-05-18 [SC] [expires: 2024-05-18]
            6113D31362A0D280FC025AAB640736427872A220
      uid                      CN=Intel(R) Software Development Products (PREPROD USE ONLY)

      pub   rsa2048 2023-08-21 [SC] [expires: 2027-08-21]
            E9BF0AFC46D6E8B7DA5882F1BAC6F0C353D04109
      uid                      CN=Intel(R) Software Development Products

3. Export GPG key to asc file:

   .. code-block:: sh

      gpg --armor --export E9BF0AFC46D6E8B7DA5882F1BAC6F0C353D04109./GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB > 53D04109.acs

   where the name of the .asc file is `the last 8 symbols <https://snapcraft.io/docs/package-repositories#:~:text=deb%2C%20deb%2Dsrc%5D-,key%2Did,-Type%3A%20string>`__

4. Save this key in ``<project>/snap/keys/folder``. Snapcraft will install the corresponding key.

5. Then, the OpenVINO apt repositoriy can be added to the snap's snapcraft.yaml by using the
   top-level package repositories keyword with a deb-type repository:

   .. code-block:: sh

      package-repositories:
        - type: apt
         components: [main]
         suites: [ubuntu20]
         key-id: E9BF0AFC46D6E8B7DA5882F1BAC6F0C353D04109
         url: https://apt.repos.intel.com/openvino/2024

6. Add OpenVINO dep packages to build-packages and stage-packages dependencies:

   .. code-block:: sh

      parts:
        app-build:
          build-packages:
            - openvino-libraries-dev
          stage-packages:
            - openvino-libraries-2024.1.0

7. Build User Application
