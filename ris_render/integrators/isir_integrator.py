import time
import os
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm, trange

mi.set_variant(os.environ.get('VARIANT', 'llvm_ad_rgb'))

from .reservoir import ChainHolder
from .utils import FastCategorical_dr, FastCategorical_np


class ISIRIntegrator(mi.SamplingIntegrator):
    # TODO: add Rao-Blackwellization (?)
    def __init__(
            self,
            props=mi.Properties(),
            n_particles: int = 10,
            weight_population: bool = False,
            avg_chain: bool = False):
        """Integrator with Iterated Sampling Importance Resampling algorithm.

        Args:
            props (_type_, optional): Legacy properties. Defaults to mi.Properties().
            n_particles (int, optional): Size of proposals population. Defaults to 10.
            weight_last_population (bool, optional): Whether use weighted estimate with population
                from last step of i-SIR. The estimate is unbiased. Defaults to False.
            avg_chain (bool, optional): Whether use the estimate with population along the
                chain. The estimate is unbiased when using weighted estimate from each step (i.e.
                weight_last_population=True). Defaults to False.
                
        Example:
            props = mi.Properties()
            props['emitter_samples'] = 10
            props['bsdf_samples'] = 0

            integrator = ISIRIntegrator(
                props, n_particles=10, weight_population=True, avg_chain=True)

        """
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        # self.check_visibility = props.get('check_visibility', True)
        # self.hide_emitters = props.get('hide_emitters', False)
        # self.num_resamples = props.get('num_resamples', 10)

        warnings.warn("BSDF sampler is not implemented, setting 'bsdf_samples' to 0.")
        self.bsdf_samples = 0
        assert self.emitter_samples + self.bsdf_samples != 0, "Number of samples must be > 0"
        self.ttl_samples = self.emitter_samples + self.bsdf_samples
        self.m_weight_bsdf = 1. / mi.Float(self.bsdf_samples)
        self.m_weight_em = 1. / mi.Float(self.emitter_samples)
        self.m_frac_bsdf = self.m_weight_bsdf / self.ttl_samples
        self.m_frac_em = self.m_weight_em / self.ttl_samples
        
        self.n_particles =  n_particles
        if (avg_chain and not weight_population):
            warnings.warn("The estimate may be biased if chain is not stationary, consider setting 'weight_population' to True")
            # raise NotImplementedError
        
        self.avg_chain = avg_chain
        self.weight_population = weight_population

    def __render(self: mi.SamplingIntegrator,
                 scene: mi.Scene,
                 sensor: mi.Sensor,
                 seed: int = 0,
                 spp: int = 1,
                 develop: bool = True,
                 evaluate: bool = True,
                 **kwargs
                 ) -> mi.TensorXf: # TODO: implement bsdf sampler
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
        for i in trange(spp):
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

        chain_holder = ChainHolder(self.n_particles, n_history_steps=1) #self.emitter_samples - 1 if self.avg_chain else 1)
        
        # Emitter sampling
        # !!!
        # we specifically set test_visibility to False
        # as we would like to skip shadow tracing and save computations
        # and compensate via RIS
        # !!!
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        
        # ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), False, active_em)
        # shape = ds.numpy_.shape()

        cat_dist = FastCategorical_dr(self.n_particles, sampler.wavefront_size())
        ones = dr.ones(mi.Float, dr.width(sampler.wavefront_size()))
        for step in trange(self.emitter_samples):
            # we need to sample lights and get sampling weights with corresponding sample
            # ds.pdf - light sampling pdf based on properties of emitters
            # weight_em - 3d-valued light function / ds.pdf
            start = 0 if step == 0 else 1

            for particle in range(start, self.n_particles):
                ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), False, active_em)
                wo = si.to_local(ds.d)
                active_em_ris = active_em
                active_em_ris &= dr.neq(ds.pdf, 0.0)
                
                proposal_density = mi.Float(1) # only implemented for emmiter sampling, p(x) \propto L_e
                
                bsdf_val_em, _ = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em_ris)
                weight = (weight_em * bsdf_val_em).sum_()
                
                # weight = dr.select(dr.eq(weight, 0.0), ones, weight)
                
                chain_holder[(particle == 0, particle)] = (wo, ds.p, bsdf_val_em, weight, ds.pdf, active_em_ris)
                chain_holder.weight_sum += weight
                chain_holder.counter += mi.Int(1)
                
            # sample an index of proposal 
            # to accept
            
            # numpy version: stack weights in a vector
            # weights_np = chain_holder.dict['weight'][-1].numpy()
            # weights_np = weights_np.reshape(self.n_particles, -1).T

            idx = cat_dist.sample(chain_holder.weight_cumsum)
            
            idx = mi.UInt32(idx)
            arange = dr.arange(mi.UInt32, 0, dr.width(idx))
            idx = dr.width(idx) * idx + arange
            
            # next we evaluate the remaining part of integrated function
            # i.e. the bsdf part
            
            #active_em_ris = active_em
            #active_em_ris &= dr.neq(ds_pdf, 0.0)
            #bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em_ris)

            chosen_particle = []
            for value in chain_holder.dict.values():
                chosen_particle.append(dr.gather(type(value[-1]), value[-1], idx))

            # sampling density p = ds.pdf
            # integrated function (and desirable unnormalized sampling density) phat= Light * bsdf_val
            # weight_em = Light / ds.pdf
            # i.e. resampling weight = phat / p = weight_em * bsdf_val

            # update reservoir based on current sample
            # as target func as 3d-valued we simply sum over spatial axis
            # authors do the same, but use weighted sum
            
            if self.avg_chain or (step == self.emitter_samples - 1):
                partition = chain_holder.weight_sum / chain_holder.counter
                
                if self.weight_population:
                    particles = [chain_holder[(-1, particle)] for particle in range(self.n_particles)]
                else:
                    particles = [chosen_particle]
                
                denom = mi.Int(self.emitter_samples) if self.avg_chain else mi.Int(1)
                
                proposal_density = mi.Float(1)
                
                if self.weight_population:
                    ids = dr.arange(mi.UInt32, start=self.n_particles - 1, stop=dr.width(chain_holder.weight_cumsum), step=self.n_particles)
                    weight_sum = dr.gather(mi.Float, chain_holder.weight_cumsum, ids)
                    weight_sum = dr.select(dr.eq(weight_sum, 0.0), ones, weight_sum)
                    
                    importance_weight = proposal_density * partition / weight_sum / denom
                
                for particle in particles:
                    sample, final_point, final_bsdf_val, weight, pdf_val, activity_mask = particle
                    
                    weight = dr.select(dr.eq(weight, 0.0), ones, weight)

                    # if self.weight_population:
                    #     #np.sum(chain_holder.dict['weight'][-(self.emitter_samples - step + 1)])
                    #     weight_sum = mi.Float(weight_sum)
                    #     importance_weight = proposal_density * partition / weight_sum / denom
                    # else:
                    if not self.weight_population:
                        importance_weight = proposal_density * partition / weight / denom
                    
                    sampled_ray = si.spawn_ray(si.to_world(sample))
                    si_fin = scene.ray_intersect(sampled_ray, )

                    activity_mask = activity_mask & si_fin.is_valid()
                    final_emitter_val = si_fin.emitter(scene).eval(si_fin, activity_mask) / pdf_val

                    L += dr.select(activity_mask, final_emitter_val * final_bsdf_val * importance_weight, mi.Color3f(0))
                      
            chain_holder[(True, 0)] = tuple(chosen_particle)    
                
        return (L, active, [])


if __name__ == '__main__':
    mi.register_integrator("isir_direct", lambda props: ISIRIntegrator(props))
    simple_box = mi.cornell_box()

    simple_box['integrator'] = {
        'type': 'isir_direct',
        'emitter_samples': 10,
        'bsdf_samples': 1
    }

    scene_ris = mi.load_dict(simple_box)

    start = time.perf_counter()
    img_ris = mi.render(scene_ris, spp=10)
    elapsed_ris = time.perf_counter() - start
    print(elapsed_ris)
    plt.imshow(img_ris ** (1. / 2.2))
    plt.axis("off")
    plt.savefig("test.png")
