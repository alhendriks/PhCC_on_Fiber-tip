This python script is based on Legume (https://github.com/fancompute/legume and https://legume.readthedocs.io/en/latest/) which is the work of Momchil Minkov and colleagues from Shanhui Fan's group of Stanford. If you use their work they probably appreciate being cited on this article https://pubs.acs.org/doi/10.1021/acsphotonics.0c00327 . 

We (Fiore group from Eindhoven Univeristy of Technology) have used their script to optimize a Photonic Crystal Cavity (PhCC) which is on top of a fiber-tip to be used as a single ultrafine particle (UFP) sensor. We try to optimize the following values: the Q-factor, the coupling efficiency to a fiber, and the mode volume.

The script is slightly altered from the original. The main difference is the objective function where we couple the PhCC with the small NA (=0.14) of a single mode fiber (SMF).
