import taichi as ti
import argparse
import numpy as np

VULKAN = 'vulkan'
CUDA = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--arch',
                        type=str,
                        help='Arch, `vulkan` or `cuda`',
                        default=VULKAN)
    parser.add_argument('-f',
                        '--frames',
                        type=int,
                        help='Frames to run',
                        default=1000000)
    parser.add_argument('--show-gui',
                        action='store_true',
                        help='Show GUI')
    args = parser.parse_args()
    print(f'cmd args: {args}')
    return args


args = parse_args()

arch = ti.vulkan
if args.arch.lower() == CUDA:
    arch = ti.cuda
ti.init(arch=arch)


n = 320
res = (n * 2, n)
pixels = ti.field(dtype=float, shape=res)


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def paint(t: float):
    for i, j in pixels:  # Parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


if args.show_gui:
  gui = ti.GUI("Julia Set", res=(n * 2, n))

img = np.ascontiguousarray(
    np.zeros(res + (4, ), np.float32))
for i in range(args.frames):
    paint(i * 0.03)
    p_np = pixels.to_numpy()

    if args.show_gui:
        # pass
        # from taichi.lang.meta import tensor_to_image
        # tensor_to_image(pixels, img)
        # ti.sync()
        # gui.set_image(pixels)
        gui.set_image(p_np)
        gui.show()
    # import time
    # if i % 500 == 0:
    #     print('sleep for a while...')
    #     time.sleep(2)
ti.print_profile_info()
