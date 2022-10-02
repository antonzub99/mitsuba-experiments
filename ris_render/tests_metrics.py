import drjit as dr
import mitsuba as mi

import os
import argparse
import yaml
import time
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
        integrator_type = self.config.get('integrator')

        pool_size = self.config.get('pool_size', [32])
        emitter_only = self.config.get('emitter_only', True)
        for pool in pool_size:
            cur_props = mi.Properties()
            if emitter_only:
                cur_props['emitter_samples'] = pool
            else:
                cur_props['shading_samples'] = pool // 2

            integrator = getattr(integrators, integrator_type)(cur_props)
            self.integrators.append(integrator)

        scene_path = self.config.get('scene', None)
        if scene_path is not None:
            self.scene = mi.load_file(scene_path)
        else:
            self.scene = mi.load_dict(mi.cornell_box())

    def run(self):
        shadow_rays = self.config.get('shadow_rays', [1] * len(self.integrators))
        outputs = []
        for integrator, spp in zip(self.integrators, shadow_rays):
            start = time.perf_counter()
            rendered_image = mi.render(self.scene,
                                       integrator=integrator,
                                       spp=spp)
            elapsed = time.perf_counter() - start
            outputs.append({'image': rendered_image,
                            'time': elapsed})

        return outputs


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/ris_test.yaml')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    opts = parser.parse_args()

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

    config = load_config(opts.config)

    name = config['integrator']
    pool_size = config.get('pool_size')
    shadow_rays = config.get('shadow_rays')
    scene_name = config.get('scene', '../scenes/cornell_box/tmp').split('/')[2]

    runner = Runner(config)
    outputs = runner.run()

    num_pics = len(outputs)

    fig, axs = plt.subplots(figsize=(10 * num_pics, 10), ncols=num_pics, nrows=1)
    for idx in range(num_pics):
        axs[idx].imshow(outputs[idx]['image'] ** (1. / 2.2))
        seconds = outputs[idx]['time']
        axs[idx].set_title(f'{name}, M={pool_size[idx]}, N={shadow_rays[idx]}, T={seconds:.4f} sec')
        axs[idx].axis('off')
    plt.savefig(opts.output_dir+f'/{name}_{scene_name}.png')


if __name__ == '__main__':
    test()
