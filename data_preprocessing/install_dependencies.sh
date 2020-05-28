#!/bin/bash
# Check for super user priviledges.
if [ $(whoami) != root ]
then
  echo "You must be root to install the dependencies."
  echo "Please rerun the script using sudo."
  exit
fi
SCRIPT_PATH=$(realpath -s $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
# These are things that can be installed using apt.
echo -e "\e[32mInstalling General APT Dependencies\e[39m"
apt install cmake libgoogle-glog-dev libboost-all-dev libjsoncpp-dev libyaml-cpp-dev

# Install Eigen
echo -e "\e[32mDownloading Eigen\e[39m"
cd $THIRD_PARTY_DIR
EIGEN_TAR=eigen.tar.gz
curl -o $EIGEN_TAR https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar zxf $EIGEN_TAR
mv eigen-3.3.7 eigen
rm $EIGEN_TAR

# Install Ceres
echo -e "\e[32mInstalling Ceres\e[39m"
cd $THIRD_PARTY_DIR
CERES_TAR=ceres.tar.gz
curl -o $CERES_TAR http://ceres-solver.org/ceres-solver-1.14.0.tar.gz
tar zxf ceres.tar.gz
mkdir -p ceres
cd ceres
cmake ../ceres-solver-1.14.0
make -j $(nproc)
make install
cd ..
mv ceres-solver-1.14.0 ceres/ceres-solver-1.14.0
rm $CERES_TAR