import taichi as ti

ti.init(ti.cpu)

N = 8
x = ti.field(ti.f32, shape=N)

@ti.kernel
def foo_x():
    for i in x:
        x[i] = i

foo_x()

y = ti.field(ti.f32, shape=N)

@ti.kernel
def foo_y():
    for i in y:
        y[i] = x[i] * 2.0

foo_y()

print(y.to_numpy())
print(x.to_numpy())