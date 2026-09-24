# Upstream provenance

- Adapted nuScenes devkit:
  [`rst-tu-dortmund/nuscenes-devkit`](https://github.com/rst-tu-dortmund/nuscenes-devkit)
  at `e4818ff92714d28aefa3ffe0e07dacbd7f64e840`.
- Poly-MOT: retained tracker architecture, lifecycle, association, and baseline
  motion models; see the original project and license.
- motmetrics: derived from the public
  [`cheind/py-motmetrics`](https://github.com/cheind/py-motmetrics) project.
  The exact paper snapshot is commit
  `34b060ed3526811921f334cab8df96e8ac4139ec`; its repository tree is
  `d7b1799b8963028ba5f29530cf4cbf07cbe7451c`, and the vendored `motmetrics/`
  package tree is `ea70e03d58a1c0c3f8d5f005f6fa65b8ce4709ef`. The package is vendored in
  `src/motmetrics`; its license and upstream README are in
  `third_party/motmetrics`.
