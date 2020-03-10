In order to run this program, proper installation of ALE is required. No other dependencies needed. (Assuming numpy is installed per standard installation of ALE)

To run the program, place the python file in the ALE master folder (this is the folder containing CMake files and a folder named ale_py, and place ms_pacman.bin into the same folder.

By default, the program uses seed 123, frame skip of 5 and epsilon greedy as exploration function. It also uses all features by default. All of these can be easily changed in the setting parameters section
of the code, starting from line 22.

The seeds used for testing are randomly generated; although this can be adjusted, I would suggest against doing so, as random seeds best evaluate the degree of generalization of the agent's learnt model
If desired, however, this seed can be hard-coded in line 397