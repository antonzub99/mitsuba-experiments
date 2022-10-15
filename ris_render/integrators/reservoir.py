from typing import TypeVar, Callable, Union

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
        #self.emitter_val = mi.Vector3f(0)
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
            #emitter_val: mi.Vector3f,
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
        #previous_light = self.emitter_val
        previous_bsdf = self.bsdf_val
        previous_activity_mask = self.activity_mask
        previous_weight = self.weight
        previous_pdf_value = self.pdf_val
        replace_threshold = weight / self.weight_sum

        replace_prob = self.rng.next_float32()
        active = replace_prob < replace_threshold
        self.sample = dr.select(active, sample, previous_sample)
        self.final_point = dr.select(active, final_point, previous_point)
        #self.emitter_val = dr.select(active, emitter_val, previous_light)
        self.bsdf_val = dr.select(active, bsdf_val, previous_bsdf)
        self.weight = dr.select(active, weight, previous_weight)
        self.pdf_val = dr.select(active, pdf_value, previous_pdf_value)
        self.activity_mask = dr.select(active, activity_mask, previous_activity_mask)


class ReservoirISIR(Reservoir):
    def __init__(self, n_proposals, **kwargs):
        super().__init__(**kwargs)
        self.n_proposals = n_proposals

    def update(self,
               sample: mi.Vector3f,
               final_point: mi.Vector3f,
               bsdf_val: mi.Vector3f,
               weight: Union[mi.Float, mi.Vector3f],
               pdf_value: mi.Float,
               activity_mask: mi.Bool,
               ):
        self.weight_sum += weight
        self.samples_count += mi.Int(1)

        previous_sample = self.sample
        previous_point = self.final_point
        previous_bsdf = self.bsdf_val
        previous_activity_mask = self.activity_mask
        previous_weight = self.weight
        previous_pdf_value = self.pdf_val
        #replace_threshold = weight / self.weight_sum

        replace_prob = self.rng.next_float32()
        active = replace_prob < 1.0 # replace_threshold
        # active = dr.all(dr.neq(sample, previous_sample))

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
    res = iter(reservoirs)
    loop = mi.Loop("Reservoir combining", lambda: (res, output))
    while loop((reservoir := next(res, "End")) != "End"):
        output.update(
            sample=reservoir.sample,
            bsdf_val=reservoir.bsdf_val,
            weight=reservoir.weight,
            pdf_value=reservoir.pdf_val,
            activity_mask=reservoir.activity_mask
        )
    output.current_weight = output.weight_sum * dr.rcp(output.pdf_val) * dr.rcp(output.samples_count)
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
            radius: int
    ):
        self.num_iterations = num_iterations
        self.random_indices = random_xy(num_iterations, radius)

    def __call__(self, image_of_reservoirs):
        """
        Assuming image_of_reservoirs is H x W
        """
        h, w = image_of_reservoirs.shape[0:2]
        for iteration in range(self.num_iterations):
            add_ind = self.random_indices[iteration]
            for i in range(h):
                for j in range(w):
                    new_ind = (i + add_ind[0], j + add_ind[1])
                    if new_ind[0] < 0 or new_ind[0] >= h:
                        continue
                    if new_ind[1] < 0 or new_ind[1] >= w:
                        continue
                    if i == 39 and j == 79:
                        print(image_of_reservoirs[39,79].weight_sum)
                    image_of_reservoirs[i,j] = combine_reservoirs_([image_of_reservoirs[i, j], image_of_reservoirs[new_ind]])
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
