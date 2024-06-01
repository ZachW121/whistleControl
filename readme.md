The program uses a highpass filter and gausian filter to process the incoming signal as well as a 
hard cutoff for frequencies above human ranges. From their whistle gestures
are split into 3 segments from the original 2 second gesture starting at the first peak, 
then each section is averaged out and assigned a value low, mid, or high.

There are two different implementations of how low, mid, and high values are assigned
one is main.py that assigns them relative to the other frequencies, and the other is tmp.py that 
uses preset ranges to assign them.

To run the program you are required to have python 3.12 or greater and poetry the python extension installed.
Navigate into the whistleControl directory and type in the terminal poetry init to install the required packages.
Following this you can run the tmp.py and main.py by using the command `poetry run [file.py]`