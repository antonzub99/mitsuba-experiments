__all__ = ['RISIntegrator',
           'SeqRISIntegrator']

import drjit as dr
import mitsuba as mi
import time
import matplotlib.pyplot as plt

from typing import Union

from ris_render.integrators import MyBaseIntegrator
from ris_render.integrators import Reservoir

mi.set_variant('cuda_ad_rgb')


class RISIntegrator(MyBaseIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        # self.check_visibility = props.get('check_visibility', True)
        # self.hide_emitters = props.get('hide_emitters', False)
        # self.num_resamples = props.get('num_resamples', 10)

        assert self.emitter_samples + self.bsdf_samples != 0, "Number of samples must be > 0"
        self.ttl_samples = self.emitter_samples + self.bsdf_samples
        self.m_weight_bsdf = 1. / mi.Float(self.bsdf_samples)
        self.m_weight_em = 1. / mi.Float(self.emitter_samples)
        self.m_frac_bsdf = self.m_weight_bsdf / self.ttl_samples
        self.m_frac_em = self.m_weight_em / self.ttl_samples

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

        bsdf_ctx = mi.BSDFContext()

        # ----------- general parameters ----------
        ray = mi.Ray3f(ray)  # redundant conversion just in case
        L = mi.Spectrum(0)  # radiance accumulationq
        si: mi.SurfaceInteraction3f = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=mi.Bool(True), active=active)
        active = active & si.is_valid()

        # Directly visible emission if any
        L += dr.select(active, si.emitter(scene, active).eval(si), mi.Color3f(0))

        bsdf: mi.BSDF = si.bsdf(ray)

        reservoir = Reservoir(size=sampler.wavefront_size())
        # Emitter sampling
        # !!!
        # we specifically set test_visibility to False
        # as we would like to skip shadow tracing and save computations
        # and compensate via RIS
        # !!!
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        # Streaming RIS using weighted reservoir sampling
        for idx in range(self.emitter_samples):

            # we need to sample lights and get sampling weights with corresponding sample
            # ds.pdf - light sampling pdf based on properties of emitters
            # weight_em - 3d-valued light function / ds.pdf
            ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), False, active_em)
            active_em_ris = active_em
            active_em_ris &= dr.neq(ds.pdf, 0.0)
            wo = si.to_local(ds.d)

            # next we evaluate the remaining part of integrated function
            # i.e. the bsdf part
            bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em_ris)

            # sampling density p = ds.pdf
            # integrated function (and desirable unnormalized sampling density) phat= Light * bsdf_val
            # weight_em = Light / ds.pdf
            # i.e. resampling weight = phat / p = weight_em * bsdf_val

            # update reservoir based on current sample
            # as target func as 3d-valued we simply sum over spatial axis
            # authors do the same, but use weighted sum

            reservoir.update(wo, ds.p, bsdf_val_em, (weight_em * bsdf_val_em).sum_(), ds.pdf, active_em_ris)

        # sample_np = reservoir.sample.numpy_()

        # spatial resampling / i-SIR

        reservoir_weight = reservoir.weight_sum / (reservoir.samples_count * reservoir.weight)

        sampled_ray = si.spawn_ray(si.to_world(reservoir.sample))
        si_fin = scene.ray_intersect(sampled_ray, reservoir.activity_mask)

        activity_mask = reservoir.activity_mask & si_fin.is_valid()
        final_emitter_val = si_fin.emitter(scene).eval(si_fin, activity_mask) / reservoir.pdf_val

        final_bsdf_val = reservoir.bsdf_val

        L += dr.select(activity_mask, final_emitter_val * final_bsdf_val * reservoir_weight, mi.Color3f(0))

        return (L, active, [])


class SeqRISIntegrator(MyBaseIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        self.sequence_len = props.get('sequence_len', 1)

        # self.check_visibility = props.get('check_visibility', True)
        # self.hide_emitters = props.get('hide_emitters', False)
        # self.num_resamples = props.get('num_resamples', 10)

        assert self.emitter_samples + self.bsdf_samples != 0, "Number of samples must be > 0"
        self.ttl_samples = self.emitter_samples + self.bsdf_samples
        self.m_weight_bsdf = 1. / mi.Float(self.bsdf_samples)
        self.m_weight_em = 1. / mi.Float(self.emitter_samples)
        self.m_frac_bsdf = self.m_weight_bsdf / self.ttl_samples
        self.m_frac_em = self.m_weight_em / self.ttl_samples

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

        bsdf_ctx = mi.BSDFContext()

        # ----------- general parameters ----------
        ray = mi.Ray3f(ray)  # redundant conversion just in case
        L = mi.Spectrum(0)  # radiance accumulationq
        si: mi.SurfaceInteraction3f = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=mi.Bool(True), active=active)
        active = active & si.is_valid()

        # Directly visible emission if any
        L += dr.select(active, si.emitter(scene, active).eval(si), mi.Color3f(0))

        bsdf: mi.BSDF = si.bsdf(ray)
        reservoir = Reservoir(size=sampler.wavefront_size())
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        # Streaming RIS using weighted reservoir sampling
        for idx in range(self.sequence_len):
            reservoir = self.resampling_step(scene, sampler, bsdf, bsdf_ctx, si, active_em, reservoir)
            # spatial resampling / i-SIR
            reservoir_weight = reservoir.weight_sum / (reservoir.samples_count * reservoir.weight * reservoir.pdf_val)
            estimate_weight = reservoir_weight / self.sequence_len
            sampled_ray = si.spawn_ray(si.to_world(reservoir.sample))
            si_fin = scene.ray_intersect(sampled_ray, reservoir.activity_mask)

            activity_mask = reservoir.activity_mask & si_fin.is_valid()
            final_emitter_val = si_fin.emitter(scene).eval(si_fin, activity_mask)
            final_bsdf_val = reservoir.bsdf_val
            L += dr.select(activity_mask, final_emitter_val * final_bsdf_val * estimate_weight, mi.Color3f(0))

        return (L, active, [])

    def resampling_step(self,
                        scene: mi.Scene,
                        sampler: mi.Sampler,
                        bsdf: mi.BSDF,
                        bsdf_ctx: mi.BSDFContext,
                        interaction: Union[mi.SurfaceInteraction3f, mi.Interaction3f],
                        mask: Union[mi.Bool, bool],
                        reservoir: Reservoir,
                        **kwargs
                        ):

        for idx in range(self.emitter_samples):
            # we need to sample lights and get sampling weights with corresponding sample
            # ds.pdf - light sampling pdf based on properties of emitters
            # weight_em - 3d-valued light function / ds.pdf
            ds, weight_em = scene.sample_emitter_direction(interaction, sampler.next_2d(), False, mask)
            active_em_ris = mask
            active_em_ris &= dr.neq(ds.pdf, 0.0)
            wo = interaction.to_local(ds.d)

            # next we evaluate the remaining part of integrated function
            # i.e. the bsdf part
            bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, interaction, wo, active_em_ris)

            # sampling density p = ds.pdf
            # integrated function (and desirable unnormalized sampling density) phat= Light * bsdf_val
            # weight_em = Light / ds.pdf
            # i.e. resampling weight = phat / p = weight_em * bsdf_val

            # update reservoir based on current sample
            # as target func as 3d-valued we simply sum over spatial axis
            # authors do the same, but use weighted sum
            reservoir.update(wo, ds.p, bsdf_val_em, (weight_em * bsdf_val_em).sum_(), ds.pdf, active_em_ris)
        return reservoir


if __name__ == '__main__':
    mi.register_integrator("my_dir", lambda props: RISIntegrator(props))
    simple_box = mi.cornell_box()

    simple_box['integrator'] = {
        'type': 'my_dir',
        'emitter_samples': 10,
        'bsdf_samples': 1,
    }

    scene_ris = mi.load_dict(simple_box)

    start = time.perf_counter()
    img_ris = mi.render(scene_ris, spp=1)
    elapsed_ris = time.perf_counter() - start
    print(elapsed_ris)
    plt.imshow(img_ris ** (1. / 2.2))
    plt.axis("off")
    plt.savefig("myris.png")
    #plt.show()
