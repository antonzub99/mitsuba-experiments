import drjit as dr
import mitsuba as mi


def mis_weight(pdf_a, pdf_b):
    """MIS with power heuristic."""
    
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


class RISIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        #self.hide_emitters = props.get('hide_emitters', False)
        
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
        
        #main rendering loop here - process each sample here
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
            
        #block.put(sample_pos, value=rgb)
        
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
        
        #loop = mi.Loop(name="Emitter sampling", state=lambda: (sampler, si, active_em, L))
        #loop.set_max_iterations(self.emitter_samples)

        for idx in range(self.emitter_samples): 
        
            
            ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            # emitter MIS 
            wo = si.to_local(ds.d)
            bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf * self.m_frac_em, bsdf_pdf_em * self.m_frac_bsdf) * self.m_weight_em)
            #mis_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf_em))
            L += dr.select(active_em, weight_em * bsdf_val_em * mis_em, mi.Color3f(0))
       
        # BSDF sampling
        active_bsdf = active
        """
        loop2 = mi.Loop(name="BSDF sampling", state=lambda: (
            sampler, si, active_bsdf, L))
        loop2.set_max_iterations(self.bsdf_samples)
        print("set bsdf loop")
        while loop2(active_bsdf):
        """ 
        for idx in range(self.bsdf_samples):
            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(active_bsdf), sampler.next_2d(active_bsdf), active_bsdf)
            ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))

            active_bsdf &= dr.any(dr.neq(bsdf_weight, 0.0))
            
            si_bsdf = scene.ray_intersect(ray_bsdf, active_bsdf)
            L_bsdf = si_bsdf.emitter(scene).eval(si_bsdf, active_bsdf)
        
            # BSDF MIS
            ds = mi.DirectionSample3f(scene, si_bsdf, si)
            delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            emitter_pdf = scene.pdf_emitter_direction(si, ds, active_bsdf & ~delta)
            mis_bsdf = mis_weight(bsdf_sample.pdf * self.m_frac_bsdf, emitter_pdf * self.m_frac_em) * self.m_weight_bsdf
            #mis_bsdf = mis_weight(bsdf_sample.pdf, emitter_pdf)
            L += dr.select(active_bsdf, L_bsdf * bsdf_weight * mis_bsdf, mi.Color3f(0))
        return (L, active, [])   
    