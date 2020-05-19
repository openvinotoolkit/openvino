# nGraph-ONNX Continuous Integration Script
The proper script running nGraph-ONNX tests can be found in ngraph-onnx repository:
https://github.com/NervanaSystems/ngraph-onnx/tree/master/.ci/jenkins/ci.groovy

Jenkinsfile in this directory just downloads and runs CI stored in repository mentioned above.
This is due to how Jenkins Multibranch Pipeline jobs are implemented, which don't provide an option to automatically clone different repository than the one for which the build is triggered.

# MANUAL REPRODUCTION INSTRUCTION
From directory containing CI scripts execute runCI.sh bash script:

```
cd <path-to-repo>/.ci/onnx/jenkins/
./runCI.sh
```

To remove all items created during script execution (files, directories, docker images and containers), run:

```
./runCI.sh --cleanup
```

After first run, executing the script will rerun tox tests. To rebuild nGraph and run tests use:

```
./runCI.sh --rebuild
```
