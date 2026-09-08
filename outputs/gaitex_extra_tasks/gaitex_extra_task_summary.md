# GAITEX Extra Task MTU Validation Summary

This analysis extends the paired IMU versus optical GAITEX validation beyond natural gait.
It includes natural gait (`ng`), gait with obstacle (`gwo`), and instructed movement variants (`rd`, `rgs`).

## Run Summary

- Completed segments: 257 / 257
- Subjects represented: 19
- Overall median key joint correlation: 0.945
- Overall median MTU waveform correlation: 0.908
- Overall median MTU normalized RMSE: 33.2% of IMU derived ROM

## Task Level Summary

| Task | Segments | Subjects | Key joint corr | MTU corr | MTU NRMSE (%) | MTU ROM (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gwo | 54 | 18 | 0.747 | 0.702 | 45.5 | 15.5 |
| ng | 55 | 18 | 0.926 | 0.879 | 23.4 | 24.4 |
| rd | 76 | 19 | 0.934 | 0.913 | 83.2 | 4.6 |
| rgs | 72 | 18 | 0.989 | 0.975 | 17.4 | 35.0 |

## Segment Label Summary

| Task | Label | Segments | Subjects | Key joint corr | MTU corr | MTU NRMSE (%) | MTU ROM (mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gwo | gwo_v0 | 18 | 18 | 0.763 | 0.710 | 54.3 | 6.1 |
| gwo | gwo_v1 | 18 | 18 | 0.715 | 0.690 | 40.5 | 20.0 |
| gwo | gwo_v2 | 18 | 18 | 0.752 | 0.700 | 40.1 | 21.4 |
| ng | ng_v0 | 18 | 18 | 0.847 | 0.812 | 58.0 | 5.8 |
| ng | ng_v1 | 18 | 18 | 0.958 | 0.910 | 16.3 | 29.2 |
| ng | ng_v2 | 18 | 18 | 0.935 | 0.905 | 18.2 | 29.8 |
| ng | ng_v3 | 1 | 1 | 0.864 | 0.846 | 16.5 | 39.1 |
| rd | rd_correct | 19 | 19 | 0.953 | 0.939 | 65.3 | 5.2 |
| rd | rd_pronation | 19 | 19 | 0.946 | 0.913 | 78.0 | 5.5 |
| rd | rd_supination | 19 | 19 | 0.920 | 0.903 | 108.3 | 4.0 |
| rd | rd_toes | 19 | 19 | 0.916 | 0.885 | 85.9 | 3.7 |
| rgs | rgs_abduction | 18 | 18 | 0.984 | 0.972 | 19.2 | 31.5 |
| rgs | rgs_correct | 18 | 18 | 0.986 | 0.962 | 17.3 | 27.9 |
| rgs | rgs_flexion | 18 | 18 | 0.992 | 0.979 | 17.7 | 36.4 |
| rgs | rgs_stork | 18 | 18 | 0.995 | 0.981 | 16.9 | 41.0 |

## Interpretation

The results support use of GAITEX as a controlled robustness extension rather than as a clinical pathology cohort.
The additional tasks test whether IMU driven MTU trajectories remain comparable with optical motion capture during obstacle gait and instructed movement variants.
The data should be described as healthy participants performing controlled task variants, not as patient validation.
