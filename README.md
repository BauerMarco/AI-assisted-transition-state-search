# AI-assisted-transition-state-search

To the best of my knowledge there is not a single packaged transition state guesser using AI, so in order to use this notebook you have to do the following, before starting up the notebook. Get the tsdiff code

```
git clone --depth 1 https://github.com/seonghann/tsdiff
```

and export it to your PYTHONPATH, e.g.

```
export PYTHONPATH=$PYTHONPATH:/home/usr/tsdiff
```

then create your conda environment as shown [here](https://github.com/seonghann/tsdiff) and install py3Dmol as

```
conda install py3dmol -c conda-forge
```

This environment contains a lot of dependencies of which some are quite old. For smooth execution this course is therefore split into two notebooks, and you will have to set up a different environment for the introductory notebook as follows

```
pip install torch torchvision torchaudio
pip install torchani ase py3Dmol jupyter
conda install xtb-python -c conda-forge
```

If you don't have an nvidia GPU, you obviously don't need to install the corresponding cuda extensions.

Beware, that due to the pretrained models, the repo is rather large.
