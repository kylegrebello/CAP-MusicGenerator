import py_midicsv
import glob
import os
import shutil



for file in glob.glob('Output/*.mid.txt'):
	with open(file, 'r') as inf:

		wholeInput = inf.read()

		splitCommands = str(wholeInput).split('\', \'')
		#print(splitCommands)

		outputCommands = []

		finalLevel = len(splitCommands)
		currentLevel = 1

		for st in splitCommands:


			tupleList = st.split(',')
			

			print(tupleList)

			outputList = []
			if(len(tupleList) > 5):
				#outputList.append(tupleList[1])
				outputList.append(tupleList[2])
				outputList.append(tupleList[4])
			elif(len(tupleList) == 3):
				outputList.append(tupleList[0])
				outputList.append(tupleList[1])
				outputList.append(tupleList[2])

				
				

			outputCommands.append(outputList)
			currentLevel = currentLevel + 1




	with open(file + 'clean.txt', 'w') as of:
		of.write(str(outputCommands))
		