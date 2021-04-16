import numpy as np
import taichi as ti

real = ti.f32
# arch = ti.x64
arch = ti.metal
print_mtl = False
ti.init(default_fp=real, arch=arch, print_kernel_llvm_ir=print_mtl)
ti.set_logging_level(ti.DEBUG)
# grid parameters
N = 128

# setup sparse simulation data arrays
r = ti.field(dtype=ti.f32)
# s = ti.field(dtype=real)  # storage for reductions

rp = ti.root.pointer(ti.ijk, [N // 4])
rp.dense(ti.ijk, 4).place(r)

# ti.root.place(s)


@ti.kernel
def init():
    for i, j, k in ti.ndrange(N, N, N):
        r[i, j, k] = 1.0

@ti.kernel
def reduce(p: ti.template(), q: ti.template()):
    s = 0.0
    for I in ti.grouped(p):
        s += p[I] * q[I]
        # print(s)
    print('sum=', s)

def main():
  init()
  # return

  # s[None] = 0.0
  reduce(r, r)
  # initial_rTr = s[None]

  print(rp.num_dynamically_allocated)
  # print(f'initial_rTr={initial_rTr}')

if __name__ == '__main__':
  main()