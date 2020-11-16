#!usr/bin/bash
conda create -n fbl python=3.6 -y
conda activate fbl
cd ..
git clone https://github.com/rtqichen/ffjord.git
cd FlowBasedLikelihoods
pip install -r requirements.txt
