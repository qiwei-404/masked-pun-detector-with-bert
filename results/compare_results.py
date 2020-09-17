import sys
import pandas as pd
import numpy as np

class ProcessResults:

    def __init__(self, resultfile, testfile):
        self._predict_from_prob(resultfile)
        self._extract_true_label(testfile)

    def _predict_from_prob(self, resultfile):
        self.pred = []
        self.neg = []
        self.pos = []
        with open(resultfile, "r") as f_in:
            for line in f_in:
                probabilities = [float(prob) for prob in line.split("\t")]
                prediction = -1
                if probabilities[0] < probabilities[1]:
                    prediction = 1
                elif probabilities[0] > probabilities[1]:
                    prediction = 0
                self.pred.append(prediction)
                self.neg.append(probabilities[0])
                self.pos.append(probabilities[1])
            f_in.close()

    def _extract_true_label(self, testfile):
        self.true = []
        with open(testfile, "r") as f_in:
            for (i, line) in enumerate(f_in):
                if i == 0:
                    continue
                label = int(line.split("\t")[2])
                self.true.append(label)
            f_in.close()

    def write_results(self, outfile):
        with open(outfile, "w") as f_out:
            for i in range(len(self.pred)):
                row = "\t".join(str(prob) for prob in [self.neg[i], self.pos[i], self.pred[i], self.true[i]])
                row += "\n"
                f_out.write(row)
            f_out.close()


class SummarizeResults:

    def __init__(self, resultfile):
        self.results = pd.read_csv(resultfile, header=None, sep='\t')

    def wrong_prediction(self, outfile):
        with open(outfile, "w") as f:
            for index, row in self.results.iterrows():
                if row[2] != row[3]:
                    f.write(str(index+1)+"\n")
            f.close()


if __name__ == "__main__":
    resultfile = sys.argv[1]
    testfile = sys.argv[2]
    total_results = sys.argv[3]
    wrong_pred = sys.argv[4]
    result_processor = ProcessResults(resultfile, testfile)
    result_processor.write_results(total_results)
    summarizer = SummarizeResults(total_results)
    summarizer.wrong_prediction(wrong_pred)

