# mitsuba-experiments

`ris_render` has notebook to test integrator modules and `.py` files with custom integrators

`mitsuba-tutorials` is just a fork from tutorial repo (but I didn't set up submodules, so you can't access it from this repo)

pt - path tracer (don't need it)

mis - path tracer with mis (don't need it)

direct - simple direct illumination with 2 rays per sample per pixel (low quality)

direct_new - reimplementation of original `direct.cpp` (but there are some shadow artifacts)

resampling will be included in direct_new module

Scenes are downloaded from https://mitsuba.readthedocs.io/en/latest/src/gallery.html (but not all of them are in this repo
