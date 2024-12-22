"Subdomain Uncertainty Optimization for Cross-Speed Fault Diagnosis" has been accepted for presentation at ICASSP 2025.

I would like to express my sincere gratitude to Mr. Zhang Tairui for his invaluable assistance in completing the experiments. My heartfelt thanks also go to the other collaborators for their support and contributions.

If you want to use this code, please
- install python(3.9.16), pytorch(2.0.1, py3.9_cuda11.7_cudnn8_0), and numpy, scipy, sklearn, etc.
- download cwru and jnu datasets and put them into the [`datasets`](datasets) folder.
- run `python t.py -m SUO -t 1 -ds cwru -src 0 1 0 2 0 3 1 2 1 3 2 3 -tar 1 0 2 0 3 0 2 1 3 1 3 2` in cwru, and `python t.py -m SUO -t 1 -ds jnu -src 600 600 800 800 1000 1000 -tar 800 1000 600 1000 600 800` in jnu.