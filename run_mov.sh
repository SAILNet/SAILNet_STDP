#Movie Data and Keep Spikes
python SAILNet.py dW_SAILnet None -n 300 --plot_interval 100 --OC1 6 -s 5000 -d 30000 -k True -v True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -v True
#Movie Data, Static Learning
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -j True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -j True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -v True -j True
#Movie Data, Keep Spikes and Decay W
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -w True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -w True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -v True -w True
#Movie Data, Static Learning and W Decay
python SAILNet.py dW_SAILnet None -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -w True -j True
python SAILNet.py dW_time_dep Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -d 30000 -k True -v True -w True -j True
#python SAILNet.py dW_time_dep Border_Gaussian -n 30000 --plot_interval 10000 --OC1 6 -s 5000 -b 30000 -k True -v True -w True -j True
