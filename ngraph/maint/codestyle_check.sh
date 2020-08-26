cd ~/openvino/bin/intel64/Release/lib/
wget -nc https://github.com/google/google-java-format/releases/download/google-java-format-1.9/google-java-format-1.9-all-deps.jar

java -jar google-java-format-1.9-all-deps.jar --set-exit-if-changed |
     -r ~/openvino/inference-engine/ie_bridges/java/tests/BlobTests.java ~/openvino/inference-engine/ie_bridges/java/tests/BlobTests.java