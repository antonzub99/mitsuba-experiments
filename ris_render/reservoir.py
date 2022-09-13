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
        self.weight = mi.Float(0)
        self.pdf_val = mi.Float(0)
        self.bsdf_val = mi.Vector3f(0)
        self.activity_mask = mi.Bool(True)

        self.weight_sum = mi.Float(0)
        self.samples_count = mi.Int(0)

        self.rng = mi.PCG32(size=size)

    def update(
            self,
            #sample: mi.Ray3f,
            sample: mi.Vector3f,
            bsdf_val: mi.Vector3f,
            weight: mi.Float,
            pdf_value: mi.Float,
            activity_mask: mi.Bool
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
        previous_bsdf = self.bsdf_val
        previous_activity_mask = self.activity_mask
        previous_weight = self.weight
        previous_pdf_value = self.pdf_val
        replace_threshold = weight / self.weight_sum

        replace_prob = self.rng.next_float32()
        active = replace_prob < replace_threshold
        self.sample = dr.select(active, sample, previous_sample)
        self.bsdf_val = dr.select(active, bsdf_val, previous_bsdf)
        self.weight = dr.select(active, weight, previous_weight)
        self.pdf_val = dr.select(active, pdf_value, previous_pdf_value)
        self.activity_mask = dr.select(active, activity_mask, previous_activity_mask)


#___________________________________________________
# doesn't work below this line
#___________________________________________________


def combine_reservoirs(*reservoirs):
    output = Reservoir(next(iter(reservoirs)).size)
    for reservoir in reservoirs:
        output.update(
            sample=reservoir.current_sample,
            weight=reservoir.current_pdf_value * reservoir.current_weight * reservoir.samples_count
        )
    output.current_weight = output.weight_sum / output.current_pdf_value / output.samples_count
    return output


class SpatialReuseFunctor:
    def __init__(
            self,
            num_iterations: int,
            neighbors_indices_extractor: Callable
    ):
        self.num_iterations = num_iterations
        self.neighbors_indices_extractor = neighbors_indices_extractor

    def __call__(self, image_of_reservoirs):
        output = type(image_of_reservoirs)()
        for iteration in range(self.num_iterations):
            pixel_indices = ...  # todo
            neighbor_indices = self.neighbors_indices_extractor(pixel_indices)
            output = combine_reservoirs(*output, *image_of_reservoirs[neighbor_indices])
        return output
