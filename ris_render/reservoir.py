from typing import TypeVar, Callable

import mitsuba as mi

Sample = TypeVar('Sample')


class Reservoir:
    def __init__(self, size: int):
        current_sample= Sample()
        current_weight = mi.Float([0] * size)
        weight_sum = mi.Float([0] * size)
        samples_count = mi.Int([0] * size)
        current_pdf_value = mi.Float([0] * size)

    def update(
            self,
            sample: Sample,
            weight: mi.Float
    ):
        self.weight_sum += weight
        self.samples_count += 1
        replace_threshold = weight / self.weight_sum
        # masked update of current samples # todo


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
