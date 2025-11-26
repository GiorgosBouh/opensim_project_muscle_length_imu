                  ┌─────────────────────────────┐
                  │  NONAN IMU Dataset (CSV)    │
                  └───────────────┬─────────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ csv_to_gait2392_mot.py │
                     │  Convert IMU → .mot    │
                     └──────────────┬─────────┘
                                    │
                                    ▼
                        ┌────────────────────┐
                        │ run_muscle_lengths │
                        │  OpenSim Gait2392  │
                        │  → muscle lengths  │
                        └─────────┬──────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ gait_cycle_muscle_lengths  │
                    │  cycle detection + 0–100%  │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                        ┌───────────────────────┐
                        │ plot_muscles / all    │
                        │ mean + SD + ROM plots │
                        └─────────────┬─────────┘
                                      │
                                      ▼
                            ┌────────────────┐
                            │ Results/Report │
                            └────────────────┘
							
Predicting Lower Limb Muscle Length from IMU Gait Data

(Proof-of-concept pipeline using NONAN GaitPrint dataset + OpenSim)

This repository provides a simple and reproducible pipeline for estimating lower-limb muscle–tendon lengths using only IMU-based joint kinematic data and the Gait2392 musculoskeletal model in OpenSim.

It demonstrates that muscle lengths can be derived without:
	•	optical motion capture
	•	force plates
	•	laboratory-grade equipment

The method only requires IMU joint-angle data.

⸻

1. Dataset Used

We use publicly available IMU gait data from:

Wiles et al. (2025),
NONAN GaitPrint: An IMU gait database of healthy older adults
https://springernature.figshare.com/articles/dataset/NONAN_GaitPrint_An_IMU_gait_database_of_healthy_older_adults/27815034

The dataset contains:
	•	41 healthy older adults
	•	full-body IMU walking trials
	•	overground walking on a 200 m track
	•	spatiotemporal variables + raw kinematics

For this proof-of-concept we selected:
	•	Subject S135: 3 walking trials
	•	Subject S146: 3 walking trials

⸻

2. Dependencies

Required:
	•	Python 3.9+
	•	OpenSim 4.5
	•	NumPy
	•	pandas
	•	matplotlib

Install OpenSim using conda: conda create -n opensim_env python=3.9
conda activate opensim_env
conda install -c opensim-org -c conda-forge opensim
pip install numpy pandas matplotlib

3. Files Required

Download and place these in the working folder:S135/S135_G03_D01_B01_T0X.csv
S146/S146_G03_D01_B01_T0X.csv
gait2392_simbody.osim

4. Scripts and What They Do

inspect_trial.py

Used to check a trial:
	•	prints column names
	•	checks duration and sampling

run_muscle_lengths.py   (core step)

This is where OpenSim is used.
For every frame of IMU data it:
	•	loads Gait2392 model
	•	applies joint angles
	•	calculates muscle–tendon length

Outputs muscle lengths for 6 key muscles:
	•	gluteus medius
	•	rectus femoris
	•	vastus lateralis
	•	tibialis anterior
	•	soleus
	•	medial gastrocnemius

gait_cycle_muscle_lengths.py

Takes the long time-series and:
	•	identifies heel strikes
	•	splits the signal into gait cycles
	•	normalizes each cycle to 0–100%

plot_muscle_lengths.py

Plots IMU→OpenSim muscle lengths for one trial:
	•	mean curve
	•	variability bands
	•	heatmap

plot_all_subjects.py

Compares all 6 trials:
	•	subject-wise mean ± SD curves
	•	S135 vs S146 comparison plots

Not used in this analysis
	•	run_static_optimization.py (requires GRF)
	•	batch_csv_to_mot.py (optional utility)

⸻

5. Full Pipeline (Step-by-Step)

Step 1 — Extract muscle length:

For each trial:python run_muscle_lengths.py model.osim TRIAL.csv TRIAL_muscles.csv

Example: python run_muscle_lengths.py \
  gait2392_simbody.osim \
  S135/S135_G03_D01_B01_T01.csv \
  S135_G03_D01_B01_T01_muscles.csv

You will get a file like: S135_G03_D01_B01_T01_muscles.csv
Step 2 — Normalize into gait cycles
python gait_cycle_muscle_lengths.py \
  TRIAL.csv \
  TRIAL_muscles.csv \
  TRIAL_muscles_normcycles.csv
  For example: python gait_cycle_muscle_lengths.py \
  S135/S135_G03_D01_B01_T01.csv \
  S135_G03_D01_B01_T01_muscles.csv \
  S135_G03_D01_B01_T01_muscles_normcycles.csv
  
  Step 3 — Plot per-trial muscle behavior
  python plot_muscle_lengths.py TRIAL_normcycles.csv OUTPUT_FOLDER

  Step 4 — Compare all subjects
  python plot_all_subjects.py

  This generates:
	•	mean ± SD per subject
	•	subject comparison plots

⸻

6. Output

You will get:
	•	muscle_length CSV files
	•	gait-cycle normalized CSV files
	•	per-trial plots
	•	cross-subject comparison plots

These show:
	•	physiological muscle-length patterns across gait
	•	variability between trials
	•	differences between participants

⸻

7. Why this is useful

This pipeline demonstrates that muscle–tendon behavior can be approximated from IMU data alone, which enables:
	•	field-based analysis
	•	low-cost assessment
	•	non-laboratory biomechanics
	•	suitability for clinics, sports science, and rehabilitation

No motion capture or force plates needed.

⸻

8. Notes & Known Limitations
	•	No EMG or ground truth is included (proof-of-concept only)
	•	Real-time implementation is possible
	•	Can be easily extended to more muscles or subjects

⸻

9. Contact and Licence

For questions, file an issue or extend the scripts.
MIT License

Copyright (c) 2025 Dr. Georgios Bouchouras
https://giorgosbouh.github.io/github-portfolio/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE.
  
