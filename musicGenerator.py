import argparse
import os
import numpy
import glob
from music21 import *
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import *

parser = argparse.ArgumentParser()
parser.add_argument('--runType', type=str, help='train | predict are the only two acceptable options')
args = parser.parse_args()

notesFromCorpus = []
noteToIntegerMappingSet = []
inputTrainingSequence = []
expectedTrainingSequenceOutput = []
lengthOfTrainingSequence = 10
predictionSequenceLength = 250
lstmUnitSize = 512
denseUnitSize = 256
dropoutRate = .3

for file in glob.glob("Bach/*.mxl"):
	parsedMusicFile = converter.parse(file)

	instrumentsInScore = instrument.partitionByInstrument(parsedMusicFile)
	if instrumentsInScore is not None:
  		getFirstPartOfInstrument = instrumentsInScore.parts[0]
  		notesAndChordsInScore = getFirstPartOfInstrument.recurse()
	else:
	  notesAndChordsInScore = parsedMusicFile.flat.notes

	for scoreElement in notesAndChordsInScore:
		if isinstance(scoreElement, note.Note):
			notesFromCorpus.append(str(scoreElement.pitch))
		elif isinstance(scoreElement, chord.Chord):
			currentNote = ''
			for chordElement in scoreElement.normalOrder:
				currentNote += str(chordElement)
			chordDelimeter = '.'
			notesFromCorpus.append(chordDelimeter.join(currentNote))

lengthOfNotesFromCorpus = len(notesFromCorpus)
uniqueSetOfNotesSorted = sorted(set(notesFromCorpus))
uniqueSetOfNotesSize = len(uniqueSetOfNotesSorted)

mappedInteger = 0
for currentNote in uniqueSetOfNotesSorted:
	noteToIntegerMappingSet.append((currentNote, mappedInteger))
	mappedInteger += 1

noteToIntegerMappingDictionary = dict(noteToIntegerMappingSet)

maxRangeOfTrainingSequence = lengthOfNotesFromCorpus - lengthOfTrainingSequence
for i in range(0, maxRangeOfTrainingSequence):
	subSequenceForInputTrainingSequence = notesFromCorpus[i:i + lengthOfTrainingSequence]
	currentInputTrainingSequenceList = []
	for item in subSequenceForInputTrainingSequence:
		currentInputTrainingSequenceList.append(noteToIntegerMappingDictionary[item])
	inputTrainingSequence.append(currentInputTrainingSequenceList)
	outputSequenceIndex = notesFromCorpus[i + lengthOfTrainingSequence]
	expectedTrainingSequenceOutput.append(noteToIntegerMappingDictionary[outputSequenceIndex])

lengthOfInputTrainingSequence = len(inputTrainingSequence)
numpyReshapeRange = (lengthOfInputTrainingSequence, lengthOfTrainingSequence, 1)
if args.runType == 'train':
	inputTrainingSequence = numpy.reshape(inputTrainingSequence, numpyReshapeRange)
	inputTrainingSequence = inputTrainingSequence / float(uniqueSetOfNotesSize)
	expectedTrainingSequenceOutput = np_utils.to_categorical(expectedTrainingSequenceOutput)

	inputTrainingSequenceNormalized = numpy.reshape(inputTrainingSequence, numpyReshapeRange)
	inputTrainingSequenceNormalized = inputTrainingSequenceNormalized / float(uniqueSetOfNotesSize)

	neuralNetworkModel = Sequential()
	neuralNetworkModel.add(LSTM(lstmUnitSize, input_shape=(inputTrainingSequence.shape[1], inputTrainingSequence.shape[2]), activation="relu", return_sequences=True))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(LSTM(lstmUnitSize, activation="relu", return_sequences=True))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(LSTM(lstmUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(denseUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(denseUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(uniqueSetOfNotesSize, activation="softmax"))
	neuralNetworkModel.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	weightsOutputFilename = "weights.hdf5"
	neuralNetworkModelCheckpoint = ModelCheckpoint(
		weightsOutputFilename,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	listOfNeuralNetworkModelCallbacks = [neuralNetworkModelCheckpoint]

	neuralNetworkModel.fit(inputTrainingSequence, expectedTrainingSequenceOutput, epochs=100, batch_size=128, callbacks=listOfNeuralNetworkModelCallbacks)
elif args.runType == 'predict':
	inputTrainingSequenceNormalized = numpy.reshape(inputTrainingSequence, numpyReshapeRange)
	inputTrainingSequenceNormalized = inputTrainingSequenceNormalized / float(uniqueSetOfNotesSize)

	neuralNetworkModel = Sequential()
	neuralNetworkModel.add(LSTM(lstmUnitSize, input_shape=(inputTrainingSequenceNormalized.shape[1], inputTrainingSequenceNormalized.shape[2]), activation="relu", return_sequences=True))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(LSTM(lstmUnitSize, activation="relu", return_sequences=True))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(LSTM(lstmUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(denseUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(denseUnitSize, activation="relu"))
	neuralNetworkModel.add(BatchNormalization())
	neuralNetworkModel.add(Dropout(dropoutRate))

	neuralNetworkModel.add(Dense(uniqueSetOfNotesSize, activation="softmax"))

	neuralNetworkModel.load_weights('weights.hdf5')

	randomStart = numpy.random.randint(0, lengthOfInputTrainingSequence - 1)
	startingSequence = inputTrainingSequence[randomStart]
	listOfPredictions = []
	listOfGeneratedNotes = []
	noteOffsetFromBeginning = 0

	mappedInteger = 0
	for currentNote in uniqueSetOfNotesSorted:
		noteToIntegerMappingSet.append((mappedInteger, currentNote))
		mappedInteger += 1

	noteToIntegerMappingDictionary = dict(noteToIntegerMappingSet)

	for i in range(predictionSequenceLength):
		numpyReshapeRangeForNormalizedPredictions = (1, len(startingSequence), 1)
		listOfPossiblePredictionsNormalized = numpy.reshape(startingSequence, numpyReshapeRangeForNormalizedPredictions)
		listOfPossiblePredictionsNormalized = listOfPossiblePredictionsNormalized / float(uniqueSetOfNotesSize)

		currentPrediction = neuralNetworkModel.predict(listOfPossiblePredictionsNormalized, verbose = 0)

		noteDictionaryPredictionIndex = numpy.argmax(currentPrediction)
		predictionResult = noteToIntegerMappingDictionary[noteDictionaryPredictionIndex]
		listOfPredictions.append(predictionResult)
		startingSequence.append(noteDictionaryPredictionIndex)
		startingSequence = startingSequence[1:len(startingSequence)]

	for sequence in listOfPredictions:
		if ('.' in sequence) or sequence.isdigit():
			notesInChord = sequence.split('.')
			listOfNotesFromCurrentChord = []
			for currentNote in notesInChord:
				currentNoteFromChord = note.Note(int(currentNote))
				currentNoteFromChord.storedInstrument = instrument.Piano()
				listOfNotesFromCurrentChord.append(currentNoteFromChord)
			chordFromListOfNotes = chord.Chord(listOfNotesFromCurrentChord)
			chordFromListOfNotes.offset = noteOffsetFromBeginning
			listOfGeneratedNotes.append(chordFromListOfNotes)
		else:
			currentNote = note.Note(sequence)
			currentNote.offset = noteOffsetFromBeginning
			currentNote.storedInstrument = instrument.Piano()
			listOfGeneratedNotes.append(currentNote)

		noteOffsetFromBeginning += 0.5

	streamFromGeneratedNotes = stream.Stream(listOfGeneratedNotes)
	streamFromGeneratedNotes.write('midi', fp='test_output.mid')
