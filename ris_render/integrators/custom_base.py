__all__ = ['MyBaseIntegrator']

import drjit as dr
import mitsuba as mi
import time
import matplotlib.pyplot as plt
import numpy as np

mi.set_variant('cuda_ad_rgb')


class MyBaseIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties):
        super().__init__(props)

    def render(self: mi.SamplingIntegrator,
                 scene: mi.Scene,
                 sensor: mi.Sensor,
                 seed: int = 0,
                 spp: int = 1,
                 develop: bool = True,
                 evaluate: bool = True,
                 **kwargs
                 ) -> mi.TensorXf:
        film = sensor.film()
        sampler = sensor.sampler()

        film_size = film.crop_size()
        n_channels = film.prepare(self.aov_names())

        wavefront_size = film_size.x * film_size.y

        sampler.seed(seed, wavefront_size)

        block: mi.ImageBlock = film.create_block()
        block.set_offset(film.crop_offset())

        idx = dr.arange(mi.UInt32, wavefront_size)
        pos = mi.Vector2f()
        pos.y = idx // film_size[0]
        pos.x = idx % film_size[0]
        pos += film.crop_offset()

        aovs = [mi.Float(0)] * n_channels

        # main rendering loop here - process each sample here
        for i in range(spp):
            self.render_sample(scene, sensor, sampler, block, aovs, pos)
            sampler.advance()
            sampler.schedule_state()
            dr.eval(block.tensor())

        film.put_block(block)
        result = film.develop()
        dr.schedule(result)
        dr.eval()
        return result

    def render_sample(self,
                      scene: mi.Scene,
                      sensor: mi.Sensor,
                      sampler: mi.Sampler,
                      block: mi.ImageBlock,
                      aovs,
                      pos: mi.Vector2f,
                      active: bool = True,
                      **kwargs):
        film = sensor.film()
        scale = 1. / mi.Vector2f(film.crop_size())
        offset = - mi.Vector2f(film.crop_offset())
        sample_pos = pos + offset + sampler.next_2d()

        time = 1.
        s1, s3 = sampler.next_1d(), sampler.next_2d()
        # sensor.sample_ray performs importance sampling of the ray w.r.t sensitivity/emission profile of the endpoint
        # s1 - 1d value for spectral dimension of emission profile
        # sample_pos * scale - 2d value for sample position in pixel
        # s3 - 2d value for sample position on the aperture of the sensor

        ray, ray_weight = sensor.sample_ray(time, s1, sample_pos * scale, s3)

        medium = sensor.medium()

        active = mi.Bool(True)
        (spec, mask, aov) = self.sample(scene, sampler, ray, medium, active)
        spec = ray_weight * spec
        rgb = mi.Color3f()

        if mi.is_spectral:
            rgb = mi.spectrum_list_to_srgb(spec, ray.wavelengths, active)
        elif mi.is_monochromatic:
            rgb = spec.x
        else:
            rgb = spec

        aovs[0] = rgb.x
        aovs[1] = rgb.y
        aovs[2] = rgb.z
        aovs[3] = 1.

        block.put(sample_pos, aovs)

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium = None,
               active: mi.Bool = True,
               **kwargs):
        """
            Primary function for Monte Carlo estimation
            Here we replicate Direct Illumination pipeline
            with Resampled Importance Sampling
        """
        raise NotImplementedError
