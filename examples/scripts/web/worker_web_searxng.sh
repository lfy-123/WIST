source /path/to/your/anaconda3/etc/profile.d/conda.sh
conda activate searxng
cd /path/to/your/searxng

# First, clear the port
kill -9 $(lsof -ti:8888)

python webapp.py


