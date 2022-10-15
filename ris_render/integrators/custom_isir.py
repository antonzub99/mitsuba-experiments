__all__ = ['ISIRIntegrator']

import drjit as dr
import mitsuba as mi
import time
import matplotlib.pyplot as plt
import numpy as np

from ris_render.integrators import MyBaseIntegrator
from ris_render.integrators import ReservoirISIR

mi.set_variant('cuda_ad_rgb')


class Categorical_np:
    def __init__(self,
                 logits):
        self.probs = np.exp(logits.astype(np.float64))

        denom = np.sum(self.probs, axis=-1, keepdims=True)
        denom = np.where(denom != 0, denom, 1)
        self.probs /= denom

    def sample(self):
        return np.argmax(np.apply_along_axis(lambda x: np.random.multinomial(1, pvals=x), axis=-1, arr=self.probs), -1)


class ISIRIntegrator(MyBaseIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.shading_samples = props.get('shading_samples', 1)
        self.emitter_samples = props.get('emitter_samples', self.shading_samples)
        self.bsdf_samples = props.get('bsdf_samples', self.shading_samples)
        self.n_proposals = props.get('n_proposals', 10)
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

        reservoir = ReservoirISIR(size=sampler.wavefront_size(), n_proposals=self.n_proposals)
        # Emitter sampling
        # !!!
        # we specifically set test_visibility to False
        # as we would like to skip shadow tracing and save computations
        # and compensate via RIS
        # !!!
        active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), False, active_em)
        wo = si.to_local(ds.d)
        #bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        reservoir.update(wo, ds.p, mi.Vector3f(0), weight_em, ds.pdf, active_em)

        print('done')
        # Streaming RIS using weighted reservoir sampling
        for _k in range(self.emitter_samples):

            # we need to sample lights and get sampling weights with corresponding sample
            # ds.pdf - light sampling pdf based on properties of emitters
            # weight_em - 3d-valued light function / ds.pdf
            # proposals = [reservoir.sample]
            # proposals = [mi.DirectionSample3f()]
            # weights = [reservoir.weight]
            weights = [reservoir.weight]
            # proposals = [ds]
            wos = [reservoir.sample]

            i = mi.UInt(0)
            inner_loop = mi.Loop(name="proposals_generation", state=lambda: (sampler, i, wos, weights))
            while inner_loop(i < reservoir.n_proposals - 1):
                ds, weight_em = scene.sample_emitter_direction(si, sampler.next_2d(), False, active_em)
                #print(weight_em.sum_())
                wo = si.to_local(ds.d)
                # active_em_ris = active_em
                # active_em_ris &= dr.neq(ds.pdf, 0.0)
                # skip samples with zero weights
                # proposals.append(ds)
                # print(np.array(weight_em).shape)
                # print(weight_em.sum_())
                wos.append(wo)
                weights.append(weight_em)

                i += 1

            # proposals_np = np.stack([x.numpy_() for x in proposals])
            #for x in weights:
            #    print(np.array(x).shape)
            weights_np = np.stack([np.array(x.sum_()) for x in weights], axis=-1)
            wos_np = [np.array(wo_) for wo_ in wos]

            # weights_np = weights_np / np.sum(weights_np)
            idx = Categorical_np(np.log(weights_np)).sample()
            #ds = proposals[idx]
            active_em_ris = active_em
            active_em_ris &= dr.neq(ds.pdf, 0.0)
            #print(np.array(weights).shape)
            #print(np.stack(weights).shape)
            # make thing below work
            np_weights = np.stack(weights)
            np_wos = np.stack(wos_np)
            weight_em = np.take_along_axis(np_weights, np.broadcast_to(idx[np.newaxis, :, np.newaxis], (1,) + np_weights.shape[1:]), axis=0)
            wo = np.take_along_axis(np_wos, np.broadcast_to(idx[np.newaxis, :, np.newaxis], (1,) + np_wos.shape[1:]), axis=0)
            # next we evaluate the remaining part of integrated function
            # i.e. the bsdf part
            #print(wo.shape)
            bsdf_val_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, mi.Vector3f(wo), active_em_ris)

            # sampling density p = ds.pdf
            # integrated function (and desirable unnormalized sampling density) phat= Light * bsdf_val
            # weight_em = Light / ds.pdf
            # i.e. resampling weight = phat / p = weight_em * bsdf_val

            # update reservoir based on current sample
            # as target func as 3d-valued we simply sum over spatial axis
            # authors do the same, but use weighted sum

            reservoir.update(wo, ds.p, bsdf_val_em, mi.Vector3f(weight_em) * bsdf_val_em, ds.pdf, active_em_ris)

        # spatial resampling / i-SIR
        if isinstance(reservoir.weight, mi.Vector3f):
            res_weight = reservoir.weight.sum_()
        else:
            res_weight = reservoir.weight
        reservoir_weight = reservoir.weight_sum / (reservoir.samples_count * res_weight)

        sampled_ray = si.spawn_ray(si.to_world(reservoir.sample))
        si_fin = scene.ray_intersect(sampled_ray, reservoir.activity_mask)

        activity_mask = reservoir.activity_mask & si_fin.is_valid()
        final_emitter_val = si_fin.emitter(scene).eval(si_fin, activity_mask) / reservoir.pdf_val
        final_bsdf_val = reservoir.bsdf_val

        L += dr.select(activity_mask, final_emitter_val * final_bsdf_val * reservoir_weight, mi.Color3f(0))

        return (L, active, [])


if __name__ == '__main__':
    mi.register_integrator("my_dir", lambda props: ISIRIntegrator(props))
    simple_box = mi.cornell_box()

    simple_box['integrator'] = {
        'type': 'my_dir',
        'emitter_samples': 10,
        'bsdf_samples': 1,
        'n_proposals': 10
    }

    scene_ris = mi.load_dict(simple_box)

    start = time.perf_counter()
    img_ris = mi.render(scene_ris, spp=1)
    elapsed_ris = time.perf_counter() - start
    print(elapsed_ris)
    plt.imshow(img_ris ** (1. / 2.2))
    plt.axis("off")
    plt.savefig("myisir.png")
    #plt.show()
