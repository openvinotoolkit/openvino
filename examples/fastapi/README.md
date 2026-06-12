\# FastAPI Inference Example using OpenVINO



This example demonstrates how to run image classification inference using

FastAPI and OpenVINO Runtime.



The application is designed to be \*\*Windows-friendly\*\* and does not

automatically download large model files.



---



\## Requirements



\- Python 3.9+

\- OpenVINO Runtime

\- FastAPI

\- Uvicorn

\- OpenCV

\- NumPy



---



\## Model Setup



This example requires a MobileNet-V2 OpenVINO IR model.



\### Download the model using Open Model Zoo tools



```bash

omz\_downloader --name mobilenet-v2-pytorch

omz\_converter --name mobilenet-v2-pytorch

