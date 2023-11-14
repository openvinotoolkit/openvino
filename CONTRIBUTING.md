# Contributing to OpenVINO

## How to contribute to the OpenVINO project

OpenVINO™ is always looking for opportunities to improve and your contributions 
play a big role in this process. There are several ways you can make the 
product better:



### Provide Feedback

   * **Report bugs / issues**  
     If you experience faulty behavior in OpenVINO or its components, you can 
     [create a new issue](https://github.com/openvinotoolkit/openvino/issues)
     in the GitHub issue tracker. 

   * **Propose new features / improvements**  
     If you have a suggestion for improving OpenVINO or want to share your ideas, you can open a new
     [GitHub Discussion](https://github.com/openvinotoolkit/openvino/discussions). 
     If your idea is already well defined, you can also create a 
     [Feature Request Issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+)
     In both cases, provide a detailed description, including use cases, benefits, and potential challenges.
     If your points are especially well aligned with the product vision, they will be included in the 
     [development roadmap](./ROADMAP.md).
     User feedback is crucial for OpenVINO development and even if your input is not immediately prioritized,
     it may be used at a later time or undertaken by the community, regardless of the official roadmap.
    

### Contribute Code Changes

   * **Fix Bugs or Develop New Features**  
     If you want to help improving OpenVINO, choose one of the issues reported in 
     [GitHub Issue Tracker](https://github.com/openvinotoolkit/openvino/issues) and 
     [create a Pull Request](./CONTRIBUTING_PR.md) addressing it. Consider one of the
     tasks listed as [first-time contributions](https://github.com/openvinotoolkit/openvino/issues/17502).
     If the feature you want to develop is more complex or not well defined by the reporter, 
     it is always a good idea to [discuss it](https://github.com/openvinotoolkit/openvino/discussions) 
     with OpenVINO developers first. Before creating a new PR, check if nobody is already
     working on it. In such a case, you may still help, having aligned with the other developer.

     Importantly, always check if the change hasn't been implemented before you start working on it!
     You can build OpenVINO using the latest master branch and make sure that it still needs your
     changes. Also, do not address issues that only affect older non-LTS releases, like 2022.2.

   * **Develop a New Device Plugin**  
     Since the market of computing devices is constantly evolving, OpenVINO is always open to extending
     its support for new hardware. If you want to run inference on a device that is currently not supported, 
     you can see how to develop a new plugin for it in the 
     [Plugin Developer Guide](https://docs.openvino.ai/canonical/openvino_docs_ie_plugin_dg_overview.html).
     

### Improve documentation

   * **OpenVINO developer documentation** is contained entirely in this repository, under the 
     [./docs/dev](https://github.com/openvinotoolkit/openvino/tree/master/docs/dev) folder.

   * **User documentation** is built from several sources and published at 
     [docs.openvino.ai](docs.openvino.ai), which is the recommended place for reading 
     these documents. Use the files maintained in this repository only for editing purposes. 

   * The easiest way to help with documentation is to review it and provide feedback on the 
     existing articles. Whether you notice a mistake, see the possibility of improving the text, 
     or think more information should be added, you can reach out to any of the documentation 
     contributors to discuss the potential changes.
     
     You can also create a Pull Request directly, following the [editor's guide](./docs/CONTRIBUTING_DOCS.md).


### Promote and Support OpenVINO

   * **Popularize OpenVINO**  
     Articles, tutorials, blog posts, demos, videos, and any other involvement 
     in the OpenVINO community is always a welcome contribution. If you discuss 
     or present OpenVINO on various social platforms, you are raising awareness
     of the product among A.I. enthusiasts and enabling other people to discover
     the toolkit. Feel free to reach out to OpenVINO developers if you need help
     with making such community-based content.

   * **Help Other Community Members**  
     If you are an experienced OpenVINO user and want to help, you can always 
     share your expertise with the community. Check GitHub Discussions and 
     Issues to see if you can help someone. 


## License

By contributing to the OpenVINO project, you agree that your contributions will be 
licensed under the terms stated in the [LICENSE](./LICENSE.md) file.
