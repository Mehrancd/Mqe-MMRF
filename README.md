# Mqe-MMRF
High Accuracy Brain Image Segmentation
using this code to label brain needs to cite the paper below
Tsallis-Entropy Segmentation through MRF and Alzheimer anatomic reference for Brain Magnetic Resonance Parcellation
by: Mehran Azimbagirad et al. https://doi-org.ez67.periodicos.capes.gov.br/10.1016/j.mri.2019.11.002
This code needs a basic Python installation, with numpy, deepbrain and SimpleITK added also Docker image. We therefore used miniconda, which has Docker container available that we can inherit from: continuumio/miniconda.

In the folder containing docker file and python folder open a terminal and run the command below

docker build  -t mrbrains18/csim .

docker run --network none -dit -v output:/output:rw -v [your input folder path]:/input --name mqe_mmrf_atlas  mrbrains18/csim

docker exec mqe_mmrf_atlas python3 /mrbrains18_csim/mqe_mmrf_atlas.py        #it may takes 6 to 18 minutes

docker cp mqe_mmrf_atlas:/output [your output folder path]

docker container stop /mqe_mmrf_atlas

docker container rm  /mqe_mmrf_atlas



There are 6 parameters which can improve the results. Let me know if the results were not satisfying.
You are free to use, but you should cite in your work the original articles below:

1.	Azimbagirad, M., et al., Tsallis-Entropy Segmentation through MRF and Alzheimer anatomic reference for Brain Magnetic Resonance Parcellation. Magnetic Resonance Imaging (2019)
2.	Azimbagirad, M., et al., Partial volume transfer (PVT) conversion of cerebral tissue volumes between different magnetic fields MRI. 2019. 35(1): p. 11-20.
