# Set python path
export PYTHONPATH=$PYTHONPATH:$PWD

# Conda env
# this was set up using:
# conda create -n codi python=3.7 keras scikit-learn pandas numpy boto3 botocore opencv matplotlib
conda activate codi


# Ensure reproducible results
# From: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
export PYTHONHASHSEED=0
