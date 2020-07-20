# Inference Engine development configuration document {#openvino_docs_Inference_Engine_Development_Procedure_CONTRIBUTING}

To create MakeFiles use following process or run build-after-clone.sh script located in the root
folder if you use Ubuntu 16.04.
To create Visual Studio project run create_vs_proj_x64.cmd from scripts folder. 

## Setting up the environment for development

1. Update/init submodules bu running
```bash
git submodule init
git submodule update --recursive
```
2. Install [Git LFS](https://git-lfs.github.com) extension. It's required to download models 
   from the [repo](https://gitlab-icv.inn.intel.com/inference-engine/models-ir)
   Below is step by step guide to install Git LFS.
   
   2.1 Linux
   ```bash
    wget https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz
    tar xf git-lfs-linux-amd64-2.3.4.tar.gz
    cd git-lfs-2.3.4
    sudo PREFIX=/usr/ ./install.sh
    git config --global http.sslverify false
   ```
   2.1 Windows
        2.1.1 Download 
            [Git LFS](https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-windows-2.3.4.exe)
            and install it.
        2.1.2 Run console command 
            ```bash
            git config --global http.sslverify false
            ```
   > **NOTE**: HTTPS protocol is used to download files by Git LFS. You either have to 
   > disable HTTPS proxy for local resources like GitLab server gitlab-icv.inn.intel.com by setting 
   > `no_proxy=localhost,gitlab-icv.inn.intel.com` or switch to `http://proxy-chain.intel.com:911` proxy server, 
   > because it disables proxy for local servers automatically.

3. Use Cmake to fetch project dependencies and create Unix makefiles
   ```bash
   mkdir build
   cd build
   ```  
  There are number of options which turn on some components during builds and initiate downloading of the models

  `-DENABLE_TESTS=ON` - to build functional and behavior tests
     this will  copy necessary dependencies to ./temp folder, or to ENV.DL_SDK_TEMP folder if environment variable set
  `-DENABLE_FUNCTIONAL_TESTS=ON` - to build functional tests
  `-DCMAKE_BUILD_TYPE=Debug/Release` - to point debug or release configuration. Missing this option will generate something between
                                     Release and Debug and you might be surprised by certain aspects of the compiled binaries
  `-DENABLE_PRIVATE_MODELS=ON` - copy private models from https://gitlab-icv.inn.intel.com/inference-engine-models/private-ir with restricted access

  The full command line enough for development is following:
  ```bash
  cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
  ```

  The full command line enough for validation before push to the server
  ```bash 
  cmake -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
  ```

4. Build project and tests:
```bash
make -j16
```

5. To build documentation:
    a. Install doxygen and graphviz:
    ```bash
    apt-get install doxygen && apt-get install graphviz && apt-get install texlive
    ```
    b. Go to the documentation build directory:
    ```bash
    cd to scripts/build_documentation
    ```
    c. Run the `build_docs.sh` script:
        * To build the documentation set that includes documentation from the current branch of the 
           `inference-engine` repo and specific branches of the `openvino-documentation`, `models` and 
           `model-optimizer-tensorflow` repos, specify three branches as parameters:
       ```sh
          ./build_docs.sh ovinodoc:<OPENVINO_BRANCH> models:<MODELS_BRANCH> mo:<MO_BRANCH>
       ```
       * To build the documentation set that includes only documentation from the current branch of the
          `inference-engine` repo, run the script with no parameters:
       ```sh
          ./build_docs.sh
       ```

      > **NOTE**: You should run the script either with specifying all three parameters or without any parameters.
    
    d. Find the generated documentation in the `root_directory/doc` directory

    > **NOTE**: If you make any changes in the documentation source files, it is recommended to cleanup the 
    > documentation build directory and continue with step 3:
    >```sh
    >      cd scripts/build_documentation
    >      ./clean.sh
    >   ```
    
    > **NOTE**: The scripts for building documentation use SSH for cloning repositories. Please, make sure that
    you have
    > added your SSH key to git-lab. For more information about it, please visit the
    > [instructions page](https://gitlab-icv.inn.intel.com/help/ssh/README.md)


## Compilers supported and verified

All others may be compatible but Inference Engine does not guarantee that.

* Linux  : gcc(5.4)\*, clang(3.9)
* MacOS  : gcc(5.4), clang(3.9)\*
* Windows: MSVC(14), ICC(17.0)\*
 \* - is target compiler for platform and used for public external drops

## TeamCity CI

TeamCity CI server is available 
[here](https://teamcity01-ir.devtools.intel.com/project.html?projectId=DeepLearningSdk_DeepLearningSdk_InferenceEngine)

To get access to the server, go to 
[AGS](https://ags.intel.com/identityiq/lcm/requestAccess.jsf) and search "DevTools -- INDE xOS - Project Developer".


## Troubleshooting steps

1. **Issue**: Build of the "mkldnn" project failed on Windows with "Error MSB6003 The specified task
   executable "cmd.exe" could not be run. The working directory "\mkl\tools" does not exist".
   **Solution**: open InferenceEngine.sln -> goto "mkldnn" project 
   Properties -> Configuration Properties -> Intel Performance Libraries -> Use Intel MKL -> choose "No"