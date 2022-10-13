from typing import TypeVar, Callable

import mitsuba as mi
import drjit as dr

Sample = TypeVar('Sample')
mi.set_variant('cuda_ad_rgb')


class Reservoir:
    def __init__(self,
                 height: int = 1,
                 width: int = 1,
                 size: int = None
                 ):
        """
        mitsuba-compatible reservoir implementation
        Args:
            height: int height of the image in pixels
            width: int width of the image in pixels
            size: int (optional) total size of wavefront for parrallelization, size = height * width
        Parameters:
            current_sample: mi.Vector3f 3d vector of sampled ray direction
            current_weight: float32 resampling weight in the form of p_hat / p
                where p_hat is desirable target sampling density, p is proposal sampling density
            weight_sum: float32 running sum of resampling weights
            samples_count: int number of samples seen by the reservoir so far
            rng: random number generator, works for each pixel independently
            activity_mask: bool update mask, as each pixel is processed independently we need this for
                Monte Carlo estimation with the final sample in the reservoir

            current_pdf_value: float32 proposal sampling density value for current sample

            current_bsdf: float 32 !!!NEW!!! bsdf value along current sample
                we store this in order not to store SurfaceInteraction object

        TODO:
            current_pdf_value will be removed
            current_weight might be in mi.Vector3f format (i.e we store the value of integrated function
                and cast it from 3d to 1d be summing over spatial axis)
            the reason for that will be in readme
        """

        if size is None:
            size = height * width
        #self.current_sample = mi.Ray3f(o=mi.Vector3f(0), d=mi.Vector3f(0))
        self.sample = mi.Vector3f(0)
        self.final_point = mi.Vector3f(0)
        self.weight = mi.Float(0)
        self.pdf_val = mi.Float(0)
        self.bsdf_val = mi.Vector3f(0)
        self.activity_mask = mi.Bool(True)

        self.weight_sum = mi.Float(0)
        self.samples_count = mi.Int(0)

        self.rng = mi.PCG32(size=size)

    def update(
            self,
            sample: mi.Vector3f,
            final_point: mi.Vector3f,
            bsdf_val: mi.Vector3f,
            weight: mi.Float,
            pdf_value: mi.Float,
            activity_mask: mi.Bool,
            ):
        """
        Update function for the reservoir
        Args:
            sample: mi.Vector3f 3d vector of ray direction to the emitter
            bsdf_val: mi.Vector3f with values of bsdf function
            weight: float32 resampling weights in format p_hat / p
            pdf_value: float32 value of sampling density along given sample (will be removed)
            activity_mask: bool mask for parallel operations on pixels, is updated with other parameters

        """
        self.weight_sum += weight
        self.samples_count += mi.Int(1)

        previous_sample = self.sample
        previous_point = self.final_point
        previous_bsdf = self.bsdf_val
        previous_activity_mask = self.activity_mask
        previous_weight = self.weight
        previous_pdf_value = self.pdf_val
        replace_threshold = weight / self.weight_sum

        replace_prob = self.rng.next_float32()
        active = replace_prob < replace_threshold
        self.sample = dr.select(active, sample, previous_sample)
        self.final_point = dr.select(active, final_point, previous_point)
        self.bsdf_val = dr.select(active, bsdf_val, previous_bsdf)
        self.weight = dr.select(active, weight, previous_weight)
        self.pdf_val = dr.select(active, pdf_value, previous_pdf_value)
        self.activity_mask = dr.select(active, activity_mask, previous_activity_mask)


def combine_reservoirs(reservoirs):
    if len(reservoirs) < 1:
        return Reservoir()
    output = type(reservoirs[0])()
#     res = iter(reservoirs)
#     loop = mi.Loop("Reservoir combining", lambda: (res, output))
#     while loop((reservoir := next(res, "End")) != "End"):
    for reservoir in reservoirs:
        output.update(
            sample=reservoir.sample,
            bsdf_val=reservoir.bsdf_val,
            weight=reservoir.weight,
            pdf_value=reservoir.pdf_val,
            activity_mask=reservoir.activity_mask,
            final_point=reservoir.final_point
        )
    output.current_weight = output.weight_sum * dr.rcp(output.pdf_val) / output.samples_count
    return output

#___________________________________________________
# needs checking below this line
#___________________________________________________

def combine_reservoirs_(reservoirs):
    if len(reservoirs) < 1:
        return Reservoir()
    output = type(reservoirs[0])()
    res = iter(reservoirs)
    s_count = mi.Int(0)
    loop = mi.Loop("Reservoir combining", lambda: (res, output, s_count))
    while loop((reservoir := next(res, "End")) != "End"):
        output.update(
            sample=reservoir.sample,
            bsdf_val=reservoir.bsdf_val,
            weight=reservoir.pdf_val * reservoir.weight * reservoir.samples_count,
            pdf_val=reservoir.pdf_val,
            activity_mask=reservoir.activity_mask
        )
        s_count += reservoir.samples_count
    output.samples_count = s_count
    output.weight = output.weight_sum * dr.rcp(output.pdf_val) * dr.rcp(output.samples_count)
    return output

def random_xy(number_of_iterations, R):
    l = []
    phi2 = 1.0 / 1.3247179572447
    num = 0
    u = 0.5
    v = 0.5
    while (num < number_of_iterations * 2):
        u += phi2
        v += phi2 * phi2
        if (u >= 1.0):
            u -= 1.0
        if (v >= 1.0):
            v -= 1.0

        rSq = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5)
        if (rSq > 0.25):
            continue

        l.append(int((u - 0.5) * R))
        num += 1
        l.append(int((v - 0.5) * R))
        num += 1
    return list(zip(l[::2], l[1::2]))


class SpatialReuseFunctor:
    def __init__(
            self,
            num_iterations: int,
            radius: int,
            size: tuple
    ):
        self.num_iterations = num_iterations
        self.random_indices = random_xy(num_iterations, radius)
        self.size = size

    def __call__(self, image_of_reservoirs):
        """
        Assuming image_of_reservoirs is H x W
        """
        for iteration in range(self.num_iterations):
            add_ind = self.random_indices[iteration]
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    new_ind = (i + add_ind[0], j + add_ind[1])
                    if new_ind[0] < 0 or new_ind[0] >= self.size[0]:
                        continue
                    if new_ind[1] < 0 or new_ind[1] >= self.size[1]:
                        continue
                    i1 = i * self.size[1] + j
                    r1 = Reservoir()
                    r1.update(dr.slice(image_of_reservoirs.sample, i1),
                                            dr.slice(image_of_reservoirs.final_point, i1),
                                            dr.slice(image_of_reservoirs.bsdf_val, i1),
                                            dr.slice(image_of_reservoirs.weight, i1),
                                            dr.slice(image_of_reservoirs.pdf_val, i1),
                                            dr.slice(image_of_reservoirs.activity_mask, i1))
                    i2 = new_ind[0] * self.size[1] + new_ind[1]
                    r2 = Reservoir()
                    r2.update(dr.slice(image_of_reservoirs.sample, i2),
                                            dr.slice(image_of_reservoirs.final_point, i2),
                                            dr.slice(image_of_reservoirs.bsdf_val, i2),
                                            dr.slice(image_of_reservoirs.weight, i2),
                                            dr.slice(image_of_reservoirs.pdf_val, i2),
                                            dr.slice(image_of_reservoirs.activity_mask, i2))
                    new_r = combine_reservoirs([r1, r2])
#                     image_of_reservoirs.sample[i1] = new_r.sample
#                     image_of_reservoirs.final_point[i1] = new_r.final_point
#                     image_of_reservoirs.weight[i1] = new_r.weight
#                     image_of_reservoirs.pdf_val[i1] = new_r.pdf_val
#                     image_of_reservoirs.bsdf_val[i1] = new_r.bsdf_val
#                     image_of_reservoirs.activity_mask[i1] = new_r.activity_mask
#                     print(i1)
#                     print(new_r.sample.x[0])
#                     print(image_of_reservoirs.sample.x[i1])
                    image_of_reservoirs.sample.x[i1] = new_r.sample.x[0]
                    image_of_reservoirs.sample.y[i1] = new_r.sample.y[0]
                    image_of_reservoirs.sample.z[i1] = new_r.sample.z[0]
                    image_of_reservoirs.final_point.x[i1] = new_r.final_point.x[0]
                    image_of_reservoirs.final_point.y[i1] = new_r.final_point.y[0]
                    image_of_reservoirs.final_point.z[i1] = new_r.final_point.z[0]
                    image_of_reservoirs.bsdf_val.x[i1] = new_r.bsdf_val.x[0]
                    image_of_reservoirs.bsdf_val.y[i1] = new_r.bsdf_val.y[0]
                    image_of_reservoirs.bsdf_val.z[i1] = new_r.bsdf_val.z[0]
                    image_of_reservoirs.weight[i1] = new_r.weight[0]
                    image_of_reservoirs.pdf_val[i1] = new_r.pdf_val[0]
                    image_of_reservoirs.activity_mask[i1] = new_r.activity_mask[0]

                    image_of_reservoirs.weight_sum[i1] = new_r.weight_sum[0]
#                     print(image_of_reservoirs.samples_count)
#                     print(new_r.samples_count[0])
#                     image_of_reservoirs.samples_count = new_r.samples_count[0]
#                     dr.slice(image_of_reservoirs.weight_sum, i1) = new_r.weight_sum
#                     dr.slice(image_of_reservoirs.samples_count, i1) = new_r.samples_count
                    
        return image_of_reservoirs

# class SpatialReuseFunctor:
#     def __init__(
#             self,
#             num_iterations: int,
#             neighbors_indices_extractor: Callable
#     ):
#         self.num_iterations = num_iterations
#         self.neighbors_indices_extractor = neighbors_indices_extractor

#     def __call__(self, image_of_reservoirs):
#         output = type(image_of_reservoirs)()
#         for iteration in range(self.num_iterations):
#             pixel_indices = ...  # todo
#             neighbor_indices = self.neighbors_indices_extractor(pixel_indices)
#             output = combine_reservoirs(*output, *image_of_reservoirs[neighbor_indices])
#         return output
