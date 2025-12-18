# MARLEY_V1
Assessing Long-Term Electricity Market Design for Ambitious Decarbonization Targets using Multi-Agent Reinforcement Learning

Overview
This repository contains the implementation of a multi-agent reinforcement learning (MARL) framework for modeling long-term electricity markets. The model enables the assessment of various market designs, policy instruments, and decarbonization strategies in electricity systems, with a focus on capturing the adaptive behavior of profit-maximizing generation companies making investment decisions.
The framework employs Independent Proximal Policy Optimization (IPPO) to simulate decentralized, competitive market environments where multiple agents invest in generation assets through wholesale markets, capacity remuneration mechanisms, and Contract for Differences (CfD) auctions.
Key Features

Explicit auction mechanisms: Models long-term market designs including capacity markets and CfD auctions
Multiple policy instruments: Supports simultaneous evaluation of various policy layers (e.g., carbon taxes, support schemes)
Portfolio management: Agents manage both new investments and existing assets, capturing incumbent vs. entrant dynamics
Market competition analysis: Captures the impact of market concentration on decarbonization outcomes
Flexible market design: Enables testing of radically different institutional arrangements and hybrid market models

Repository Structure
├── environment/              # Main RLlib environment implementation
├── training/                 # Training scripts and configuration
├── evaluation/              # Scripts for evaluating trained agents
├── data/                    # Base Excel files for training scenarios
├── checkpoints/             # Exemplary trained checkpoint (CRM+CfD, 16 agents)
├── utils/                   # Utility scripts for plotting and analysis
├── alternative_envs/        # Alternative environment implementations (with penalty)
├── environment.yml          # Conda environment specification
└── README.md               # This file
Installation
Requirements

Python 3.8+
Conda or Miniconda
Linux-based system (recommended for HPC environments)

Setup

Clone this repository:

bashgit clone https://github.com/jjgonzalez2491/MARLEY_V1.git
cd MARLEY_V1

Create and activate the conda environment:

bashconda env create -f environment.yml
conda activate marley
Usage
Training
To train agents in the electricity market environment:
bashpython training/train.py [--config CONFIG_FILE]
The training script is configured for HPC environments with LSF job scheduling. Modifications may be required for local execution.
Evaluation
To evaluate trained agents:
bashpython evaluation/evaluate.py [--checkpoint CHECKPOINT_PATH]
An exemplary checkpoint for a system with 16 agents under capacity market and CfD mechanisms is provided in the checkpoints/ folder.
Data Files
Base Excel files containing market parameters, technology characteristics, and demand profiles are located in the data/ folder. These files define the stylized Italian electricity system used in the paper.
Visualization
Plotting utilities are provided in the utils/ folder for generating market outcome visualizations and performance metrics.
Model Description
The framework models generation companies (GENCOs) as independent reinforcement learning agents making investment decisions across multiple time periods. Key modeling features include:

Investment mechanisms: Capacity markets, Contract for Differences auctions, and merchant investments in the wholesale market
Decentralized learning: Independent PPO agents responding to market signals without coordination
Competitive dynamics: Captures strategic interactions between agents with varying market shares
Policy integration: Incorporates carbon pricing and other regulatory instruments
System constraints: Represents demand requirements, technology availability, and resource adequacy

The model has been validated through extensive hyperparameter searches to ensure that decentralized training yields outcomes consistent with competitive market behavior.
Alternative Implementations
The alternative_envs/ folder contains modified versions of the environment that implement the penalty mechanism discussed in the paper. These variants can be used to explore different behavioral assumptions and market clearing approaches.
Important Notes
⚠️ HPC System Requirements: Many files in this repository are designed for High-Performance Computing systems running Linux, particularly those using LSF (Load Sharing Facility) job scheduling.
⚠️ Local Execution: For simulations on personal computers with limited resources or different operating systems (Windows, macOS), major modifications will be necessary. These may include:

Adjusting parallelization settings
Modifying job submission scripts
Reducing environment complexity or agent count
Changing file paths and system-specific dependencies

Citation
If you use this framework in your research, please cite:
bibtex[Your citation information here]
Application
This framework was applied to analyze the Italian electricity system under various scenarios:

Different levels of market competition (4, 8, 16, and 32 agents)
Alternative market designs (Energy-only, Capacity markets, CfD mechanisms)
Multiple policy scenarios including carbon pricing

Results demonstrate the critical role of market design in achieving decarbonization targets while maintaining price stability and system reliability.
License
[Specify your license here]
Contact
For questions, issues, or contributions, please open an issue on GitHub or contact [your contact information].
Acknowledgments
This work is part of the STEP-CHANGE program at Politecnico di Milano, supervised by Professor Massimo Tavoni.
