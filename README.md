# Workflow
1. `greens_functions/extract_triangles.py` extracts triangles from a mesh and saves them as a .npy file.
2. `greens_functions/hundalee_greens_functions.py` computes the Greens functions for the triangles and saves them as a .npy file.
3. `inversion/prepare_inversion.py` prepares the inversion and applies relavant parameters.
4. `inversion/run_inversion.py` performs the inversion and saves the results as a .npy file.
5. `inversion/plot_results.py` plots the results of the inversion.