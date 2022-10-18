__all__ = ['MyDirectIntegrator']

import drjit as dr
import mitsuba as mi
import time
import matplotlib.pyplot as plt

from integrators import MyBaseIntegrator

mi.set_variant('cuda_ad_rgb')


def mis_weight(pdf_a, pdf_b):
    """MIS with power heuristic."""
    
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class MyDirectIntegrator(MyBaseIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        self.check_visibility = props.get('check_visibility', True)
        #self.hide_emitters = props.get('hide_emitters', False)
        
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
        
        #----------- general parameters ----------
        ray = mi.Ray3f(ray)    # redundant conversion just in case
        L = mi.Spectrum(0)     # radiance accumulation
        
        si: mi.SurfaceInteraction3f = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=mi.Bool(True), active=active)
        active = active & si.is_valid()
                
        # Directly visible emission if any
        L += dr.select(active, si.emitter(scene, active).eval(si), mi.Color3f(0))
        
        bsdf: mi.BSDF = si.bsdf(ray)
            
        # Emitter sampling
        # !!!
        # we specifically set test_visibility to False
        # as we would like to skip shadow tracing and save computations
        # and compensate via RIS
        # !!!
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        
        for idx in range(self.emitter_samples): 
                  
            ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), self.check_visibility, active_em)
            #print(f"light pdf: {ds.pdf}")
            #print(f"light value / light pdf: {weight_em}")
            #print(f"light sums: {weight_em.sum_()}")
            active_em &= dr.neq(ds.pdf, 0.0)

            # emitter MIS 
            wo = si.to_local(ds.d)
            bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            #print(f"bsdf val: {bsdf_val_em}")
            #print(f"bsdf pdf: {bsdf_pdf_em}")
            mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf * self.m_frac_em, bsdf_pdf_em * self.m_frac_bsdf) * self.m_weight_em)
            #mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf_em))
            #print(mis_em[:10])
            L += dr.select(active_em, weight_em * bsdf_val_em * mis_em, mi.Color3f(0))
            #print(f"ttl irradiance: {L}")
       
        # BSDF sampling
        active_bsdf = active
                
        for idx in range(self.bsdf_samples):
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_bsdf), sampler.next_2d(active_bsdf), active_bsdf)
            ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))

            active_bsdf &= dr.any(dr.neq(bsdf_weight, 0.0))
            
            si_bsdf = scene.ray_intersect(ray_bsdf, active_bsdf)
            active_bsdf &= si_bsdf.is_valid()
            L_bsdf = si_bsdf.emitter(scene).eval(si_bsdf, active_bsdf)
        
            # BSDF MIS
            ds = mi.DirectionSample3f(scene, si_bsdf, si)
            delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            emitter_pdf = scene.pdf_emitter_direction(si, ds, active_bsdf & ~delta)
            mis_bsdf = mis_weight(bsdf_sample.pdf * self.m_frac_bsdf, emitter_pdf * self.m_frac_em) * self.m_weight_bsdf
            #mis_bsdf = mis_weight(bsdf_sample.pdf, emitter_pdf)
            L += dr.select(active_bsdf, L_bsdf * bsdf_weight * mis_bsdf, mi.Color3f(0))
        return (L, active, [])   
    

class RISIntegrator(mi.SamplingIntegrator):
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


    def __render(self: mi.SamplingIntegrator,
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
        #

        reservoir_weight = reservoir.weight_sum / (reservoir.samples_count * reservoir.weight)

        sampled_ray = si.spawn_ray(si.to_world(reservoir.sample))
        si_fin = scene.ray_intersect(sampled_ray, reservoir.activity_mask)

        #dist = ((reservoir.final_point - si_fin.p)**2).sum_()
        #occlusion = dist > 1e-6

        activity_mask = reservoir.activity_mask & si_fin.is_valid()
        #activity_mask &= ~occlusion
        final_emitter_val = si_fin.emitter(scene).eval(si_fin, activity_mask) / reservoir.pdf_val

        final_bsdf_val = reservoir.bsdf_val

        L += dr.select(activity_mask, final_emitter_val * final_bsdf_val * reservoir_weight, mi.Color3f(0))

        # BSDF sampling

        #active_bsdf = active

        # for idx in range(self.bsdf_samples):
        #     bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_bsdf),
        #                                            sampler.next_2d(active_bsdf), active_bsdf)
        #     ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))
        #
        #     active_bsdf &= dr.any(dr.neq(bsdf_weight, 0.0))
        #
        #     si_bsdf = scene.ray_intersect(ray_bsdf, active_bsdf)
        #     active_bsdf &= si_bsdf.is_valid()
        #     L_bsdf = si_bsdf.emitter(scene).eval(si_bsdf, active_bsdf)
        #
        #     # BSDF MIS
        #     ds = mi.DirectionSample3f(scene, si_bsdf, si)
        #     delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
        #     emitter_pdf = scene.pdf_emitter_direction(si, ds, active_bsdf & ~delta)
        #     mis_bsdf = mis_weight(bsdf_sample.pdf * self.m_frac_bsdf, emitter_pdf * self.m_frac_em) * self.m_weight_bsdf
        #     # mis_bsdf = mis_weight(bsdf_sample.pdf, emitter_pdf)
        #     L += dr.select(active_bsdf, L_bsdf * bsdf_weight * mis_bsdf, mi.Color3f(0))
        return (L, active, [])


if __name__ == '__main__':
    mi.register_integrator("my_dir", lambda props: MyDirectIntegrator(props))
    simple_box = mi.cornell_box()

    simple_box['integrator'] = {
        'type': 'my_dir',
        'emitter_samples': 10,
        'bsdf_samples': 10,
    }

    scene_ris = mi.load_dict(simple_box)

    start = time.perf_counter()
    img_ris = mi.render(scene_ris, spp=1)
    elapsed_ris = time.perf_counter() - start
    print(elapsed_ris)
    plt.imshow(img_ris ** (1. / 2.2))
    plt.axis("off")
    plt.savefig("mydirect.png")
    #plt.show()
