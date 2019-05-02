#install the basics
echo 'basic upgrades'
sudo apt-get -y update
sudo apt-get -y dist-upgrade
sudo apt-get -y install wget graphviz


##to install with pip
#echo 'python3 upgrade'
sudo apt-get -y install python3 python3-dev build-essential
#echo 'pip3 install'
sudo apt-get -y install python3-pip wget graphviz

#install python packages
echo 'python package install'
sudo pip3 install graphviz pydot tqdm scipy numpy matplotlib biopython pandas

##installing tensorflow
echo 'tensorflow install'
sudo pip3 install tensorflow
