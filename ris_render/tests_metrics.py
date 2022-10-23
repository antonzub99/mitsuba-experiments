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


def make_props_list(props: dict,
                    num_exps: int):
    props_list = []
    for idx in range(num_exps):
        props_list.append(dict())

    for key, value in props.items():
        if len(value) != len(props_list):
            value = [value] * len(props_list)
        for cur_prop, val in zip(props_list, value):
            cur_prop[key] = val

    return props_list


class Runner:
    def __init__(self, config):
        self.config = config
        self.integrators = []
        render_cfg = self.config['base_render']
        integrator_type = render_cfg.get('integrator')
        self.num_exps = render_cfg.get('exps')

        props = render_cfg.get('props')
        props_list = make_props_list(props, self.num_exps)

        add_props = self.config.get('other', None)
        if add_props is not None:
            add_list = make_props_list(add_props, self.num_exps)
        else:
            add_list = [None] * self.num_exps

        for props_single, add_single in zip(props_list, add_list):
            cur_props = mi.Properties()
            for key, val in props_single.items():
                cur_props[key] = val

            if add_single is not None:
                cur_add = dict()
                for key, val in add_single.items():
                    cur_add[key] = val
                integrator = getattr(integrators, integrator_type)(props=cur_props, **cur_add)
            else:
                integrator = getattr(integrators, integrator_type)(props=cur_props)
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
    #pool_size = base_cfg.get('pool_size')
    #shadow_rays = base_cfg.get('shadow_rays')
    #sequence_len = base_cfg.get('sequence_len', [1] * len(shadow_rays))
    scene_name = config.get('scene', '../scenes/cornell_box/tmp').split('/')[2]

    runner = Runner(config)
    outputs, reference = runner.run()

    num_pics = len(outputs)

    fig, axs = plt.subplots(figsize=(10 * num_pics + 1, 10), ncols=num_pics+1, nrows=1)
    for idx in range(num_pics):
        axs[idx].imshow(outputs[idx]['image'] ** (1. / 2.2))
        seconds = outputs[idx]['time']
        metric = outputs[idx]['rmae']
        axs[idx].set_title(f'{name}, T={seconds:.4f} sec, RMAE={metric:.4f}')
        axs[idx].axis('off')
    axs[-1].imshow(reference ** (1. / 2.2))
    axs[-1].set_title(f'Reference')
    axs[-1].axis('off')
    plt.savefig(opts.output_dir+f'/{name}_{scene_name}.png')


if __name__ == '__main__':
    test()
