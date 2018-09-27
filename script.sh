apt-get update
sudo apt-get install python2.7 -y
sudo apt-get install python-pip python-dev build-essential -y 
pip install -r requirements.txt
sudo apt-get install python-scipy -y
python myfile.py
sudo apt-get install default-jre -y
echo "alias naila='python /usr/bin/naila/naila.py'" >>~/.bashrc
. ~/.bashrc
