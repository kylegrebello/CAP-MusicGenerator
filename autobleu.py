import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import glob
import argparse

def argparser():
	Argparser = argparse.ArgumentParser()
	Argparser.add_argument('--cand', type=str)
	args = Argparser.parse_args()
	return args

args = argparser()

candidate = []
scoreList = []
avgScore = 0.0
numRef = 0.0
hiScore = 0.0

smooth = SmoothingFunction().method7

with open(args.cand, 'r') as f2:
	for line in f2:
		for word in line.split('], ['):
			candidate.append(word)

for file in glob.glob('references/*.mid.txtclean.txt'):
	with open(file, 'r') as f1:
		numRef = numRef + 1
		reference = []
		for line in f1:
			for word in line.split('], ['):
				reference.append(word)
		score = nltk.translate.bleu_score.sentence_bleu([reference], candidate, smoothing_function=smooth)
		if(score == 1):
			numRef = numRef-1
		else:
			avgScore += score
			if(hiScore < score):
				hiScore = score
			scoreList.append(score)

avgScore = avgScore / numRef

print(scoreList)

print('high score is equal to ' + str(hiScore))
print('avgScore is equal to ' + str(avgScore))


