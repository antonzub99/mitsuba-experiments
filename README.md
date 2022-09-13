# mitsuba-experiments

`ris_render` has notebook to test integrator modules and `.py` files with custom integrators

`mitsuba-tutorials` is just a fork from tutorial repo (but I didn't set up submodules, so you can't access it from this repo)

pt - path tracer (don't need it)

mis - path tracer with mis (don't need it)

direct - simple direct illumination with 2 rays per sample per pixel (low quality)

direct_new - reimplementation of original `direct.cpp` (but there are some shadow artifacts)

resampling will be included in direct_new module

Scenes are downloaded from https://mitsuba.readthedocs.io/en/latest/src/gallery.html (but not all of them are in this repo

# state of work

0. mitsuba-compatible reservoir is added. Check `ris_render\reservoir.py` and docstrings there. 

    * In the best case, we only need to store ray directions of samples, their resampling weights and bool mask for parallel operations

    * mitsuba provides high-level API to trace rays, compute intersections and its properties (position of intersection, emitterptr here, bsdfptr). With this `SurfaceIntersection` instance its easy to compute everything we need for Monte-Carlo estimation

    * but it's way too expensive to store intersection info in reservoir

    * so I'll fix the working logic to trace the final winner sample an get all the function\pdf values along this direction correctly

    * if we store not just resampling weights as floats, but as 3d-valued functions (i.e. values of integrated function, conversion to target pdf is achieved by simple summation over spatial axis), then we can also store the final function\pdf value

    * we still need to trace final ray to check if its occluded

1. RIS implementation + dumping reservoir info from mitsuba to numpy format. Check `RISIintegrator` in `ris_render\custom_direct_new.py` for some info on resampling if you want to.
    
    * Anton will do that (few fixes are left)
 
2. Reservoir combination with means of mitsuba

3. Spatial resampling as in ReSTIR with numpy

4. move iSIR from pytorch to numpy

5. make sequential resampling in our pipeline (steps 1, 2, 4 must be finished for that)

6. (bonus) Spatial resampling and everything else via drjit 



