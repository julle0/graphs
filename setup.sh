echo "Creating virtual env..."
python -m venv main
echo "Hopping onto the virtual environment..."
source main/bin/activate
pip install --upgrade pip
pip install sympy numpy scipy
echo "Done."
