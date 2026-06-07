# Wrapper to run the `vae` package on a remote Colab VM via `colab exec -f`.
# This mimics `python -m vae` so that relative imports resolve correctly.
import sys
sys.path.insert(0, '/content')

import runpy
runpy.run_module('vae', run_name='__main__')
