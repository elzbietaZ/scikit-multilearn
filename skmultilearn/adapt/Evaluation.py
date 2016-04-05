import sklearn, time


class Evaluation:
    def __init__(self, classifier, rounds_count, train_set, test_set):
        self.classifier = classifier
        self.rounds_count = rounds_count
        self.train_set = train_set
        self.test_set = test_set
        self.hamming_loss = 0
        self.accuracy = 0
        self.time = 0

    def evaluate(self):
        single_hamming_loss = 0
        single_accuracy = 0
        singe_time = 0
        i = 0

        while i < self.rounds_count:
            time_start = time.clock()
            self.classifier.fit(self.train_set['X'], self.train_set['y'])
            predictions = self.classifier.predict(self.test_set['X'])
            single_hamming_loss = sklearn.metrics.hamming_loss(self.test_set['y'], predictions)
            single_accuracy = sklearn.metrics.accuracy_score(self.test_set['y'], predictions)
            print ("HL: " + str(single_hamming_loss))
            print ("AC: " + str(single_accuracy))
            self.hamming_loss += single_hamming_loss
            self.accuracy += single_accuracy
            singe_time = time.clock() - time_start
            self.time += singe_time
            i+=1
        self.hamming_loss = self.hamming_loss / self.rounds_count
        self.accuracy = self.accuracy / self.rounds_count
        print ("final HL: " + str(self.hamming_loss))
        print ("final AC: " + str(self.accuracy))
        return("final HL: " + str(self.hamming_loss)+"        "+"final AC: " + str(self.accuracy)+"     time: "+str(self.time))
