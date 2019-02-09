import numpy as np

class CharRNN(object):
    def __init__(self, corpus, hidden_size=128, seq_len=25, lr=1e-2, epochs=100):
        """
        Arguments:
            corpus {str} -- Entire text corpus
        
        Keyword Arguments:
            hidden_size {number} -- size of hidden state (default: {128})
            seq_len {number} -- time steps to unroll for (default: {25})
            lr {number} -- learning rate (default: {1e-3})
            epochs {number} -- number of epochs to train for (default: {100})
        """
        self.corpus = corpus
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs

        # create mapping from characters to numbers and back
        chars = list(set(corpus))
        self.data_size, self.input_size, self.output_size = len(corpus), len(chars), len(chars)
        self.char_to_num = {c:i for i,c in enumerate(chars)}
        self.num_to_char = {i:c for i,c in enumerate(chars)}

        self.h = np.zeros((self.hidden_size , 1))

        self.W_xh = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.W_hy = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_y = np.zeros((self.output_size, 1))

    def __loss(self, X, Y):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.h)

        # forward pass
        loss = 0
        for t in range(len(X)):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][X[t]] = 1
            hs[t] = np.tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][Y[t], 0])

        # backward pass
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        delta = np.zeros_like(hs[0])
        for t in reversed(range(len(X))):
            dy = np.copy(ps[t])
            # backprop into y
            dy[Y[t]] -= 1
            dW_hy += np.dot(dy, hs[t].T)
            db_y += dy

            # backprop into h
            dh = np.dot(self.W_hy.T, dy) + delta
            dh_raw = (1 - hs[t] ** 2) * dh
            db_h += dh_raw
            dW_hh += np.dot(dh_raw, hs[t-1].T)
            dW_xh += np.dot(dh_raw, xs[t].T)

            # update delta
            delta = np.dot(self.W_hh.T, dh_raw)
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            # gradient clipping to prevent exploding gradient
            np.clip(dparam, -5, 5, out=dparam)

        # update last hidden state for sampling
        self.h = hs[len(X) - 1]
        return loss, dW_xh, dW_hh, dW_hy, db_h, db_y

    def fit(self):
        smoothed_loss = -np.log(1. / self.input_size) * self.seq_len
        for e in range(self.epochs):
            for p in range(np.floor(self.data_size / self.seq_len).astype(np.int64)):
                # get a slice of data with length at most seq_len
                x = [self.char_to_num[c] for c in self.corpus[p * self.seq_len:(p + 1) * self.seq_len]]
                y = [self.char_to_num[c] for c in self.corpus[p * self.seq_len + 1:(p + 1) * self.seq_len + 1]]

                # compute loss and gradients
                loss, dW_xh, dW_hh, dW_hy, db_h, db_y = self.__loss(x, y)
                smoothed_loss = smoothed_loss * 0.99 + loss * 0.01
                if p % 1000 == 0: print('Epoch {0}, Iter {1}: Loss: {2:.4f}'.format(e+1, p, smoothed_loss))

                # SGD update
                for param, dparam in zip([self.W_xh, self.W_hh, self.W_hy, self.b_h, self.b_y], [dW_xh, dW_hh, dW_hy, db_h, db_y]):
                    param += -self.lr * dparam

    def sample(self, seed, n):
        """Generate text from the RNN
        
        Arguments:
            seed {str} -- character to start the sequence with
            n {int} -- length of sequence
        """
        seq = []
        h = self.h

        x = np.zeros((self.input_size, 1))
        x[self.char_to_num[seed]] = 1

        for t in range(n):
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            p = np.exp(y) / np.sum(np.exp(y))

            # sample from the distribution
            seq_t = np.random.choice(range(self.input_size), p=p.ravel())

            x = np.zeros((self.input_size, 1))
            x[seq_t] = 1
            seq.append(seq_t)
        return ''.join(self.num_to_char[num] for num in seq)

if __name__ == '__main__':
    with open('input.txt', 'r') as f:
        data = f.read()

    char_rnn = CharRNN(data, epochs=10)
    char_rnn.fit()
    print(char_rnn.sample(data[0], 100))
