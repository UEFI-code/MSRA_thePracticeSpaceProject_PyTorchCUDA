# myLinear Extension For PyTorch

This is a super Linear of Neuros.

Both CUDA and CPU are supported.

May show its advantage at heavy number of Neuros (In CUDA).

The historic debug code is \*.comemorate.\* 

Code proudly wroten with Vim😎

## How to Build & Install

You need to have CUDA toolkit and PyTorch before build, and their version must Match.

```bash
sudo apt install nvidia-cuda-toolkit
pip3 install torch==[Version]
```

If you are using Ubuntu 20.04 LTS, the [Version] is likely 1.7.0 to suit CUDA Toolkit 10.1

Microsoft recommend that [Version] is 1.5.0 and CUDA Toolkit 10.0

My code is expected to work on most Version of Environment😎

//The Debug Log file is stored in /tmp/myLinearDbg.txt, you can change this in myLinear.cpp

The Cpp Debug Log will Direct Show on Screen!

To build this extension:

```bash
python3 setup.py install
```

## Hack Build Without GPU

It is possible to trick the PyTorch to continue build this sorce even without GPU installed.

For example on PyTorch 1.7.0,

```bash
vim /usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
```

Go to line 1407, comment capability = torch.cuda.get_device_capability(), then add capability = ['7', '5+PTX'] before arch_list created.

<img width="1440" alt="image" src="https://user-images.githubusercontent.com/74940000/171839610-a7618c24-387f-4d63-a8f5-c830f4025058.png">
