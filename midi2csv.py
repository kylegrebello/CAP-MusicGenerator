import py_midicsv
import glob
import os
import shutil



for file in glob.glob('Output/*.mid'):
	csv_string = py_midicsv.midi_to_csv(file)


	with open(file + '.txt', 'w') as of:
		of.write(str(csv_string).strip('[]'))