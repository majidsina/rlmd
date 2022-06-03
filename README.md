# Reinforcement Learning in Multiplicative Domains

[![temp-rgrewal - rlmd](https://img.shields.io/static/v1?label=temp-rgrewal&message=rlmd&color=blue&logo=github)](https://github.com/temp-rgrewal/rlmd "Go to GitHub repo")
[![GitHub release](https://img.shields.io/github/release/temp-rgrewal/rlmd)](https://github.com/temp-rgrewal/rlmd/releases)
[![License](https://img.shields.io/badge/License-AGPLv3-green)](https://github.com/temp-rgrewal/rlmd/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Research encompasses several overlapping areas:
1. Peculiarities regarding use of critic loss functions, tail exponents, and shadow means,
2. Multi-step returns and replay buffer coupling in continuous action spaces,
3. Learning in multiplicative (non-ergodic) domains, maximising the time-average growth rate,
4. Applications to strongly non-Markovian financial gambles using historical prices, and
5. Designing fully autonomous self-learning guidance systems for real-time target acquisition.

Implementation using [Python](https://www.python.org) 3.10 and [PyTorch](https://pytorch.org) 1.11 with environments interfaced through [Gym](https://www.gymlibrary.ml/) 0.24.

## Key Findings
### Additive Experiments
* Critic loss evaluation using MSE is an acceptable starting point, but use of HUB, MAE, and HSC should be considered as there exists serious potential for ‘free’ performance gains depending on environment.
* Critic loss mini-batches appear to exhibit extreme kurtosis (fat tails) and so aggregation using 'empirical' arithmetic means (Monte-Carlo approach) severely underestimates the true population mean.
* Multi-step returns for continuous action spaces using TD3 and SAC is not advised due to lack of global policy maximisation across the infinite action space unlike the finite discrete case.

### Multiplicative Experiments
* The maximisation of probability-based expectations methodology universally prescribed by contemporary decision theory is inappropriate for multiplicative processes due to conflation of probabilities with payoffs.
* State-of-the-art model-free off-policy reinforcement learning algorithms that are designed to maximise expected additive rewards are modified to operate in any conceivable multiplicative environment.
* The model-free agent now fully autonomously, self-learns the actions required to maximise value through the avoidance of steep losses, represented by raising the time-average growth rate.
* Direct applications encompass any situation where percentage changes (as opposed to numerical changes) in underlying values are reported, such as financial and economic modelling.
* The theory is experimentally validated by converging to known optimal growth-maximising actions (leverages) for gambles involving coin flips, die rolls, and geometric Brownian motion.
* Cost-effective risk mitigation using extremely convex insurance safe havens is investigated where the agent develops a strategy that indisputably increases value by reducing the amount of risk taken.

### Market Experiments
* Trained agents capable of operating in extremely non-Markovian financial markets using only consecutive past returns as inputs for highly liquid USD-denominated securities modelled as zero margin CFDs.
* Historical time series simulation is entirely non-parametrically performed using shuffled holdout out-of-sample forecasting with there existing significant long-term positive correlation across each simulation.
* Performance across a myriad of complex environments is found to exhibit substantial positive skew in wealth growth, however, is highly dependent on both the historical time-period and the assets included.

### Guidance Experiments
* Created fully automated guidance systems that control the trajectory of point projectiles to targets in unbounded 3D space while under the effects of slowly varying vector fields such as wind.
* Presented novel energy efficient multi-stage actors for operation in extremely remote environments and designed countermeasure systems for intercepting projectiles.
* Experimental results are currently both pending completion and further fine tuning.

## Data Analysis
Comprehensive discussion and implications of all results are described in `Grewal-RLMD.pdf`.

The data regarding optimal leverage experiments (NumPy arrays), agent training performance (NumPy arrays), and the learned models (PyTorch parameters) have a total combined size of 5.3 GB.

The breakdown for optimal leverage experiments, additive agents, multiplicative agents, and market agents are 0.4 GB, 2.7 GB, 1.4 GB, and 0.9 GB respectively. All data is available upon request.

## Usage
Using the latest [release](https://github.com/temp-rgrewal/rlmd/releases) is recommended where we adhere to [semantic](https://semver.org/) versioning.

Binary coin flip, trinary die roll, and geometric Brownian motion (GBM) experiments pertaining to empirical optimal leverages are contained in the `lev/` directory with instructions provided inside each of the files.

Training on financial market environments requires the generation of historical data sourced from [Stooq](https://stooq.com/) using `scripts/gen_market_data.py` with customisation options regarding asset selection and dates available.

All reinforcement learning agent training is executed using `main.py` with instructions provided within the file. Upon the completion of each experiment, relevant directories within `results/` titled by the reward dynamic (additive or multiplicative or market) will be created. Inside each will exist directories for data and models with subdirectories titled by the environment name.

Final aggregated figures for all agent experiments that share common training parameters are generated using `scripts/gen_figures.py` where specific aggregation details must be input in the file.

Accessing the code involves the following commands:
```commandline
git clone https://github.com/temp-rgrewal/rlmd.git

cd rlmd
```

Install all required packages (ideally in a virtual environment) with or without dependencies using:
```commandline
pip3 install -r requirements.txt

pip3 install -r requirements--no-deps.txt
```

Optimal leverage roll experiments for a particular "gamble" are conducted with:
```commandline
python lev/gamble.py
```

Historical financial market data is sourced and aggregated using:
```commandline
python scripts/gen_market_data.py
```

Reinforcement learning agent training and evaluation is executed with:
```commandline
python main.py
```

Summary figures for agent performance are generated using:
```commandline
python scripts/gen_figures.py
```

There are also additional prerequisites for installing certain packages (without mentioning dependencies):
* `box2d-py`: Required only for training on [Box2D]( https://gym.openai.com/envs/#box2d) continuous control tasks.
* `box2d-py` and `pybullet`: A C++ compiler such as [GCC](https://gcc.gnu.org/) is necessary. On Windows, install [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/) and in the installer select the “Desktop development with C++” workload with both optional features “MSVC v143 - VS 2022 C++ x64/x86 build tools” and the “Windows 10 SDK”.
* `gym`: The interface compiler [SWIG](http://www.swig.org/) must be [installed](http://www.swig.org/Doc4.0/SWIGDocumentation.pdf) to connect C/C++ programs with scripting languages. For Linux either build from the distributed tarball or directly fetch the package from a repository. On Windows, extract the swigwin zip file and add its directory to the system PATH environment variable.
* `mpi4py`: A Message Passing Interface (MPI) library for Linux or [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467) is only required for training on the [DeepMimic](https://arxiv.org/pdf/1804.02717.pdf) environments ported to [PyBullet](https://pybullet.org)
* `pandas-datareader`: Required only when updating existing or creating new market environments by directly obtaining historical financial market data (prices and volumes).
* `torch`: Only the exact version with a compatible compute platforms should be used following the official [instructions](https://pytorch.org/get-started/locally/). Different versions can significantly reduce speed and lead to broken function/method calls.

## Tests
Comprehensive tests during compilation have been written for all user inputs.

A reduced scale test across all optimal leverage experiments is performed with:
```commandline
python tests/lev_tests.py
```

The agent learning script will also terminate if critic network backpropagation fails mainly due to the use of strong outlier-supressing critic loss functions and or divergence in particular environment state components.

An initial test for the early stability of agent training across a variety of scenarios can be conducted using:
```commandline
python tests/agent_tests.py
```

## References
* Reinforcement learning ([Szepesvári 2009](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf), [Sutton and Bartow 2018](http://incompleteideas.net/book/RLbook2020.pdf))
* Feature reinforcement learning ([Hutter 2009](https://sciendo.com/downloadpdf/journals/jagi/1/1/article-p3.pdf), [Hutter 2016](https://www.sciencedirect.com/science/article/pii/S0304397516303772), [Majeed and Hutter 2018](https://www.ijcai.org/Proceedings/2018/0353.pdf))
* Twin Delayed DDPG (TD3) ([Silver et al. 2014](http://proceedings.mlr.press/v32/silver14.pdf), [Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf), [Fujimoto et al. 2018](https://arxiv.org/pdf/1802.09477.pdf))
* Soft Actor-Critic (SAC) ([Ziebart 2010](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf), [Haarnoja et al. 2017](http://proceedings.mlr.press/v70/haarnoja17a/haarnoja17a-supp.pdf), [Haarnoja et al. 2018](https://arxiv.org/pdf/1812.05905.pdf))
* Critic loss functions from NMF ([Guan et al. 2019](https://arxiv.org/pdf/1906.00495.pdf))
* Multi-step returns and replay coupling ([Meng, Gorbet and Kulic 2020](https://arxiv.org/pdf/2006.12692.pdf), [Fedus et al. 2020](https://arxiv.org/pdf/2007.06700.pdf))
* Non-iid data and fat tails ([Fazekas and Klesov 2006](https://epubs.siam.org/doi/pdf/10.1137/S0040585X97978385), [Taleb 2009](https://www.sciencedirect.com/science/article/abs/pii/S016920700900096X), [Taleb and Sandis 2014](https://arxiv.org/pdf/1308.0a58.pdf), [Cirillo and Taleb 2016](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2016.1162908?needAccess=true), [Cirillo and Taleb 2020](https://www.nature.com/articles/s41567-020-0921-x.pdf), [Taleb 2020](https://arxiv.org/ftp/arxiv/papers/2001/2001.10488.pdf), [Lagnado and Taleb 2022](https://jai.pm-research.com/content/early/2022/02/04/jai.2022.1.157))
* Landau and Lifshitz primer on statistical mechanics, ensemble averages, entropy, and optics ([1980](https://archive.org/details/landau-and-lifshitz-physics-textbooks-series/Vol%205%20-%20Landau%2C%20Lifshitz%20-%20Statistical%20Physics%20Part%201%20%283rd%2C%201980%29), [1994]( https://archive.org/details/landau-and-lifshitz-physics-textbooks-series/Vol%202%20-%20Landau%2C%20Lifshitz%20-%20The%20classical%20theory%20of%20fields%20%284th%2C%201994%29/mode/2up))
* Kelly criterion ([Bernoulli 1738](http://risk.garven.com/wp-content/uploads/2013/09/St.-Petersburg-Paradox-Paper.pdf), [Kelly 1956](https://cpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/36891/files/2017/07/Kelly1956-1uwz47o.pdf), [Ethier 2004](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/kelly-system-maximizes-median-fortune/DD46B2432B0E251CF2CFFA9E90D31A2B), [Nekrasov 2013](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2259133))
* Multiplicative dynamics ([Peters 2011a](https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true), [Peters 2011b](https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0065), [Peters 2011c](https://arxiv.org/pdf/1110.1578.pdf), [Gigerenzer and Brighton 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3629675/pdf/mjms-19-4-006.pdf), [Peters and Gell-Mann 2016](https://aip.scitation.org/doi/pdf/10.1063/1.4940236), [Peters 2019](https://www.nature.com/articles/s41567-019-0732-0.pdf), [Peters et al. 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.00056.pdf), [Meder et al. 2020](https://arxiv.org/ftp/arxiv/papers/1906/1906.04652.pdf), [Peters and Adamou 2021](https://arxiv.org/pdf/1801.03680.pdf), [Spitznagel 2021](https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797), [Vanhoyweghen et al. 2022](https://www.nature.com/articles/s41598-022-07613-6.pdf))
* Modelling time series ([Cerqueira et al. 2017](https://ieeexplore.ieee.org/document/8259815), [de Prado 2018](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), [Cerqueira, Torgo and Mozetič 2020](https://link.springer.com/content/pdf/10.1007/s10994-020-05910-7.pdf))
* Power consumption of neural networks ([Han et al. 2015](https://proceedings.neurips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf), [García-Martín et al. 2019](https://www.sciencedirect.com/science/article/pii/S0743731518308773))

## Acknowledgements
The [Sydney Informatics Hub](https://www.sydney.edu.au/research/facilities/sydney-informatics-hub.html) and the [University of Sydney](https://www.sydney.edu.au)’s high performance computing cluster, [Artemis](https://sydney-informatics-hub.github.io/training.artemis.introhpc/01-intro), for providing the computing resources that contributed to the additive experiment results reported herein.

The base TD3 and SAC algorithms were implemented using guidance from: [DLR-RM/stable-baelines3](https://github.com/DLR-RM/stable-baselines3), [haarnoja/sac](https://github.com/haarnoja/sac), [openai/spinningup](https://github.com/openai/spinningup), [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code), [rail-berkley/softlearning](https://github.com/rail-berkeley/softlearning), [rlworkgroup/garage](https://github.com/rlworkgroup/garage), and [sfujim/TD3](https://github.com/sfujim/TD3).

If you use any of this work, please cite our results like this:
```bibtex
@misc{jsgrewal2022,
  author        = {J. S. Grewal},
  title         = {Reinforcement Learning in Multiplicative Domains},
  publisher     = {GitHub},
  journal       = {GitHub Repository},
  howpublished  = {\url{https://github.com/rajabinks/rlmd}},
  year          = {2022}
  }
```
This repository also utilises the strong copyright [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) in order to encourage open-source development of all the ideas, applications, and code enclosed within this repository.
