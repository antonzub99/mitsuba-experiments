import drjit as dr
import mitsuba as mi

import os
import argparse
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

import integrators

mi.set_variant('cuda_ad_rgb')


def load_config(path):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


class Runner:
    def __init__(self, config):
        self.config = config
        self.integrators = []
        render_cfg = self.config['base_render']
        integrator_type = render_cfg.get('integrator')

        pool_size = render_cfg.get('pool_size', [32])
        emitter_only = render_cfg.get('emitter_only', True)

        sequence_len = render_cfg.get('sequence_len', [1] * len(pool_size))
        for pool, seq_len in zip(pool_size, sequence_len):
            cur_props = mi.Properties()
            if emitter_only:
                cur_props['emitter_samples'] = pool
            else:
                cur_props['shading_samples'] = pool // 2
            cur_props['sequence_len'] = seq_len

            integrator = getattr(integrators, integrator_type)(cur_props)
            self.integrators.append(integrator)
        self.shadow_rays = render_cfg.get('shadow_rays', [1] * len(self.integrators))

        scene_path = self.config.get('scene', None)
        if scene_path is not None:
            self.scene = mi.load_file(scene_path)
        else:
            self.scene = mi.load_dict(mi.cornell_box())

    def run(self):
        scene_name = self.config.get('scene', None)
        if scene_name is None:
            scene_name = 'cornell box'
        else:
            scene_name = scene_name.split('/')[2]
        print(f"Running procedure on scene {scene_name}")
        renders = self.render()
        print(f"Tests are rendered")
        reference = self.render_reference()
        print(f"Reference is rendered")
        for render in renders:
            image = render['image']
            metric = self.evaluate_rmae(image, reference)
            render['rmae'] = metric
        print(f"Metrics are calculated")
        return renders, reference

    def render(self):
        outputs = []
        for integrator, spp in zip(self.integrators, self.shadow_rays):
            start = time.perf_counter()
            rendered_image = mi.render(self.scene,
                                       integrator=integrator,
                                       spp=spp)
            elapsed = time.perf_counter() - start
            outputs.append({'image': rendered_image,
                            'time': elapsed})

        return outputs

    def render_reference(self):
        ref_cfg = self.config['ref_render']
        ref_props = mi.Properties()
        ref_props['shading_samples'] = ref_cfg['shading_samples']
        integrator = getattr(integrators, ref_cfg['integrator'])(ref_props)
        reference = mi.render(self.scene,
                              integrator=integrator,
                              spp=ref_cfg.get('shadow_rays', 1))
        return reference

    def evaluate_rmae(self, image, reference):
        """
        runs RMAE calculation for two images
        """
        #img_np = image.numpy_()
        #ref_np = reference.numpy_()
        mae = dr.mean(dr.abs(image - reference)).numpy()[0]
        denom = dr.mean(dr.abs(reference)).numpy()[0]
        return mae / denom


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/ris_test.yaml')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    opts = parser.parse_args()

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    config = load_config(opts.config)
    base_cfg = config['base_render']
    name = base_cfg['integrator']
    pool_size = base_cfg.get('pool_size')
    shadow_rays = base_cfg.get('shadow_rays')
    sequence_len = base_cfg.get('sequence_len', [1] * len(shadow_rays))
    scene_name = config.get('scene', '../scenes/cornell_box/tmp').split('/')[2]

    runner = Runner(config)
    outputs, reference = runner.run()

    num_pics = len(outputs)

    fig, axs = plt.subplots(figsize=(10 * num_pics + 1, 10), ncols=num_pics+1, nrows=1)
    for idx in range(num_pics):
        axs[idx].imshow(outputs[idx]['image'] ** (1. / 2.2))
        seconds = outputs[idx]['time']
        metric = outputs[idx]['rmae']
        axs[idx].set_title(f'{name}, M={pool_size[idx]}, N={shadow_rays[idx]}, len={sequence_len[idx]}, T={seconds:.4f} sec, RMAE={metric:.4f}')
        axs[idx].axis('off')
    axs[-1].imshow(reference ** (1. / 2.2))
    axs[-1].set_title(f'Reference')
    axs[-1].axis('off')
    plt.savefig(opts.output_dir+f'/{name}_{scene_name}.png')


if __name__ == '__main__':
    test()
