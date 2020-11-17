#!usr/bin/bash
#conda create -n fbl python=3.6 -y
#conda activate fbl
#conda install -c conda-forge schwimmbad
#sudo apt-get install gfortran
#wget https://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz
#pip install -r requirements.txt
cd ..
#git clone https://github.com/rtqichen/ffjord.git
git clone https://github.com/AlexandreAdam/LensTools.git 
cd LensTools
python setup.py develop
cd ..
#git clone https://github.com/illustristng/illustris_python.git
cd FlowBasedLikelihoods
