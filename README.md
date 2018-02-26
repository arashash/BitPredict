# BitPredict
Predicting altcoin buy/sell startegies for each day

# Installing Dependencies
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar xvfz ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure
make
sudo make install

pip install TA-Lib
conda install -c anaconda quandl
conda install -c plotly plotly
pip install pytrends
