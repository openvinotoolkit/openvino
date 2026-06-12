# OpenVINO Release Management

The process described below outlines the approach to managing OpenVINO releases.

## Release Milestones
- Planning  
- Execution (development of new features)  
- Stabilization (Feature Freeze, Code Freeze milestones)  
- Validation  
- Distribution  

---

### Planning
This phase takes **2–4 weeks** and involves:
- Scoping the backlog  
- Prioritizing tasks  
- Analyzing requirements  
- Making commitments based on timelines defined by the release manager  

---

### Execution (Development of New Features)
- [OpenVINO Contributing Guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING.md)  
- [Code Contribution Guide](https://docs.openvino.ai/2026/about-openvino/contributing/code-contribution-guide.html)  
- [OpenVINO First Good Issue](https://github.com/openvinotoolkit/openvino/issues/17502)  

---

### Stabilization (Feature Freeze, Code Freeze Milestones)

- **Feature Freeze:**  
  No new features are added after this point. The team focuses on stabilizing existing features, fixing bugs, and improving performance.

- **Code Freeze:**  
  Only critical bug fixes are allowed. This ensures stability and minimizes the risk of introducing new issues.

---

### Release Validation

- Continuous process with different cadences:
  - Nightly  
  - Bi-weekly  
  - Weekly  

- After Code Freeze, final regression testing ensures:
  - No new bugs are introduced  
  - Software meets required quality standards  

---

### Distribution

- OpenVINO supports different types of builds:
  - Regular releases  
  - Long-Term Support (LTS)  
  - Pre-release versions  
  - Nightly builds  

- Learn more:
  - [OpenVINO Release Policy](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/release-policy.html)  
  - [OpenVINO Download](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)  
