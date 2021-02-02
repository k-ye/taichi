import taichi as ti


def ti_support_dynamic(test):
    return ti.archs_excluding(ti.cc)(test)


def ti_support_non_top_dynamic(test):
    return ti.archs_excluding(ti.opengl, ti.cc)(test)


@ti_support_dynamic
def test_dynamic():
    x = ti.field(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        pass

    for i in range(n):
        x[i] = i

    for i in range(n):
        assert x[i] == i


@ti_support_dynamic
def test_dynamic2():
    x = ti.field(ti.f32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i

    func()

    for i in range(n):
        assert x[i] == i


@ti_support_dynamic
def test_dynamic_matrix():
    x = ti.Matrix.field(2, 1, dtype=ti.i32)
    n = 8192

    ti.root.dynamic(ti.i, n, chunk_size=128).place(x)

    @ti.kernel
    def func():
        ti.serialize()
        for i in range(n // 4):
            x[i * 4][1, 0] = i

    func()

    for i in range(n // 4):
        a = x[i * 4][1, 0]
        assert a == i
        if i + 1 < n // 4:
            b = x[i * 4 + 1][1, 0]
            assert b == 0


@ti_support_dynamic
def test_append():
    x = ti.field(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    elements = []
    for i in range(n):
        elements.append(x[i])
    elements.sort()
    for i in range(n):
        assert elements[i] == i


@ti_support_dynamic
def test_length():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32, shape=())
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            ti.append(x.parent(), [], i)

    func()

    @ti.kernel
    def get_len():
        y[None] = ti.length(x.parent(), [])

    get_len()

    assert y[None] == n


@ti_support_dynamic
def test_append_ret_value():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    z = ti.field(ti.i32)
    n = 128

    ti.root.dynamic(ti.i, n, 32).place(x)
    ti.root.dynamic(ti.i, n, 32).place(y)
    ti.root.dynamic(ti.i, n, 32).place(z)

    @ti.kernel
    def func():
        for i in range(n):
            u = ti.append(x.parent(), [], i)
            y[u] = i + 1
            z[u] = i + 3

    func()

    for i in range(n):
        assert x[i] + 1 == y[i]
        assert x[i] + 3 == z[i]


@ti_support_non_top_dynamic
def test_dense_dynamic():
    # 1. It appears that <= CUDA 11.1 has a weird bug, the end result
    # being that appending to Taichi's dynamic node messes up its length. See
    # https://stackoverflow.com/questions/65995357/cuda-spinlock-implementation-with-independent-thread-scheduling-supported
    # 2. Unfortunately, even if I fall back to the warp lock impl, i.e.
    # https://github.com/taichi-dev/taichi/blob/d1061750485a7f31bb3b0f824182f1c497c70051/taichi/runtime/llvm/locked_task.h#L29
    # the test still failed :( So lock is *a* problem. We need to see if 11.2
    # can solve them all.
    # 3. 11010 maps to '11.1'. The minor version comprises the
    # last three digits. See CUDA's deviceQuery example:
    # https://github.com/NVIDIA/cuda-samples/blob/b882fa00ee7151134cd40b6ef01a5a9af8fe8fa9/Samples/deviceQuery/deviceQuery.cpp#L106-L108
    if ti.cfg.arch == ti.cuda and ti.core.query_int64(
            'cuda_runtime_version') <= 11010:
        return
    n = 128
    x = ti.field(ti.i32)
    l = ti.field(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 8).place(x)

    @ti.kernel
    def func():
        ti.serialize()
        for i in range(n):
            for j in range(n):
                ti.append(x.parent(), j, i)

        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == n


@ti_support_non_top_dynamic
def test_dense_dynamic_len():
    n = 128
    x = ti.field(ti.i32)
    l = ti.field(ti.i32, shape=n)

    ti.root.dense(ti.i, n).dynamic(ti.j, n, 32).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            l[i] = ti.length(x.parent(), i)

    func()

    for i in range(n):
        assert l[i] == 0


@ti_support_dynamic
def test_dynamic_activate():
    ti.init(arch=ti.metal)
    # record the lengths
    l = ti.field(ti.i32, 3)
    x = ti.field(ti.i32)
    xp = ti.root.dynamic(ti.i, 32, 32)
    xp.place(x)

    m = 5

    @ti.kernel
    def func():
        for i in range(m):
            ti.append(xp, [], i)
        l[0] = ti.length(xp, [])
        x[20] = 42
        l[1] = ti.length(xp, [])
        x[10] = 43
        l[2] = ti.length(xp, [])

    func()
    l = l.to_numpy()
    assert l[0] == m
    assert l[1] == 21
    assert l[2] == 21
