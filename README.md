# Hybrid-random-forest-models
The model in this repository was used in an academic paper titled "Hybrid random forest models optimized by Sparrow search algorithm (SSA) and Harris hawk optimization algorithm (HHO) for slope stability prediction" This repository contains MATLAB code for implementing hybrid random forest algorithms, including SSA-RF and HHO-RF. Additionally, the underlying random forest model and all models mentioned in the paper are mentioned in this repository and can be viewed on request.
Prerequisites To run the code in this repository, you need:

• MATLAB (2021 or higher)

Installation

1. Clone this repository to your local machine.
2. Open MATLAB and navigate to the repository folder

Usage

It includes the HHO-RF folder or the SSA-RF folder. The usage methods of the two mixed forest models are basically the same. Load HHO-RF or SSA-RF into MATLAB's working path (data is the complete data if you are interested in slope stability).
1. Open the "main.m" file in MATLAB
2. The input features are slope height, slope angle, unit weight, cohesion, internal friction angle, pore pressure ratio. The main response variable is slope state.
3. Running the "main.m" file in MATLAB
4. Results, including evaluation metrics and confusion matrices used to compare model performance, are displayed in the MATLAB console

Contact Information If you have any questions or concerns, please contact Shaofeng Wang at sf.wang@csu.edu.cn.
