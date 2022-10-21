import mitsuba as mi
import drjit as dr


def dr_concat(a, b):
    size_a = dr.width(a)
    size_b = dr.width(b)
    c = dr.empty(type(a), size_a + size_b)
    dr.scatter(c, a, dr.arange(mi.UInt32, size_a))
    dr.scatter(c, b, size_a + dr.arange(mi.UInt32, size_b))
    return c
    