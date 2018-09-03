import numpy as np
from mcts import mcts
from tqdm import trange


class coaching():
    def __init__(self, game, nnet, mcts1):
        self.game = game
        self.nnet = nnet
        #self.pnet = self.nnet.__class__(self.game)
        self.mcts = mcts1
        self.prints = False

    def executeepisode(self):
        trainexample = []
        board = self.game.startphan()
        self.curplayer = 1
        episodestep = 0

        while True:
            episodestep += 1
            oneminusone = self.game.oneminusone(board, self.curplayer)
            pi = self.mcts.getactionprob(oneminusone)
            sym = self.game.symme(oneminusone, pi)
            for b, p in sym:
                trainexample.append([b, self.curplayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curplayer = self.game.nextstate(board, self.curplayer, action)
            if self.prints == True:
                print("episode :", episodestep, "\n", board)
            r = self.game.ggeutnam(board, self.curplayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curplayer))) for x in trainexample]

    def learn(self):

        for iter in range(30):
            iterationtrainexample = []
            finalexample = []
            self.prints = False
            try:
                for i in trange(10):
                    #print("game:", i)
                    if iter % 10 == 9 and i == 9:
                        self.prints = True
                    iterationtrainexample += self.executeepisode()
                for e in iterationtrainexample:
                    finalexample.append(e)
                print(finalexample)
                self.nnet.train(finalexample)
                self.mcts = mcts(self.game, self.nnet)

            except Exception as err:
                print(err)
                self.nnet.saving("~/", "model_err.ckpt")

        self.nnet.saving("~/", "model1.ckpt")








