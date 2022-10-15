__all__ = ['MyDirectIntegrator']

import drjit as dr
import mitsuba as mi
import time
import matplotlib.pyplot as plt

from ris_render.integrators import MyBaseIntegrator

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
