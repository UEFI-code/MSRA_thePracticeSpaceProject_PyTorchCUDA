# myLinear extensen For PyTorch

This is a easy Linear layer CUDA extense for PyTorch

## How to Build & Install

You need to have CUDA toolkit and PyTorch before build, and their version must Match.

```bash
sudo apt install nvidia-cuda-toolkit
pip3 install torch==[Version]
```

If you are using Ubuntu 20.04 LTS, the [Version] likely is 1.7.0 to suit CUDA Toolkit 10.1

The Debug Log file is stored in /tmp/myLinearDbg.txt, you can change this in myLinear.cpp

And then, you may run

```bash
python3 setup.py
```
