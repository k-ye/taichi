# Taichi Vulkan Backend Setup

Right now the Vulkan backend + GLSL → SPIR-V codegen is in the [vulkan-branch]. We also have a native SPIR-V codegen, which is not publicly available yet.

## Build Steps

Here the build steps are given on the Ubuntu OS. Other system should follow a similar approach for building the [vulkan-branch].

1. Follow https://vulkan-tutorial.com/Development_environment#page_Linux to install the Vulkan packages.

```sh
$ sudo apt install vulkan-tools
$ sudo apt install libvulkan-dev
$ sudo apt install vulkan-validationlayers-dev spirv-tools
```

2. Install Google's `shaderc` library. This library is for compiling GLSL to SPIR-V. Note that once we switch to the native SPIR-V codege, this step can be removed.

    Go to https://github.com/google/shaderc/blob/main/downloads.md and download the `Linux - clang` version (`gcc` should also work, but it's larger). Unarchive it. The path to the unarchived directory is assumed to be `/path/to/shaderc-clang-linux`.


3. Follow https://docs.taichi.graphics/docs/lang/articles/contribution/dev_install to build Taichi on your own. Here the `cmake` command line needs to be extended:

```sh
$ cmake .. -DCMAKE_CXX_COMPILER=clang++ -DTI_WITH_VULKAN=ON -DSHADERC_ROOT_DIR=/path/to/shaderc-clang-linux/install
```

4. If built successfully, you should be able to run the sample code in `vk_examples/`. For example:

```sh
# Inside 'taichi' repo
$ python3 vk_examples/mpm88.py
```

[vulkan-branch]: https://github.com/k-ye/taichi/tree/vk-stable-compact